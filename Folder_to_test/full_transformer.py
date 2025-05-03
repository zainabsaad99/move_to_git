import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import random
import optuna
from tqdm import tqdm
import os
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Preparation
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['Output'].str.len() <= 500]
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)
    return df['Input'].tolist(), df['Output'].tolist()

# Tokenization and Vocabulary
class Vocabulary:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        self._build_vocab()

    def _build_vocab(self):
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        all_chars = list(string.printable)
        self.char2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.char2idx.update({char: idx + len(special_tokens) for idx, char in enumerate(all_chars)})
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def __len__(self):
        return len(self.char2idx)

    def encode(self, text):
        return [self.char2idx.get(char, self.unk_token) for char in text]

    def decode(self, indices):
        return ''.join([self.idx2char.get(idx, '') for idx in indices if idx not in {self.pad_token, self.sos_token, self.eos_token}])

# Dataset Class
class CipherDataset(data.Dataset):
    def __init__(self, inputs, outputs, vocab, max_length):
        self.inputs = inputs
        self.outputs = outputs
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = str(self.inputs[idx])
        output_text = str(self.outputs[idx])
        input_encoded = [self.vocab.sos_token] + self.vocab.encode(input_text) + [self.vocab.eos_token]
        output_encoded = [self.vocab.sos_token] + self.vocab.encode(output_text) + [self.vocab.eos_token]
        input_padded = input_encoded + [self.vocab.pad_token] * (self.max_length - len(input_encoded))
        output_padded = output_encoded + [self.vocab.pad_token] * (self.max_length - len(output_encoded))
        return torch.tensor(input_padded[:self.max_length]), torch.tensor(output_padded[:self.max_length])

# Transformer Model Components
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, H, L, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ff(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        return self.norm3(x + self.dropout(self.ff(x)))

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_length)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_emb = self.dropout(self.pos_enc(self.encoder_embedding(src)))
        tgt_emb = self.dropout(self.pos_enc(self.decoder_embedding(tgt)))
        for layer in self.enc_layers:
            src_emb = layer(src_emb, src_mask)
        for layer in self.dec_layers:
            tgt_emb = layer(tgt_emb, src_emb, src_mask, tgt_mask)
        return self.fc(tgt_emb)

# Training/Evaluation Utilities
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(loader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        out = model(src, tgt[:, :-1])
        loss = criterion(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(loader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt[:, :-1])
            loss = criterion(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

# Data Preparation for Training
inputs, outputs = load_data('train_augmented.csv')
vocab = Vocabulary()
max_length = 512
train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
train_dataset = CipherDataset(train_inputs, train_outputs, vocab, max_length)
val_dataset = CipherDataset(val_inputs, val_outputs, vocab, max_length)

# Global best tracking
best_overall_model = None
best_overall_loss = float('inf')
best_config = None

def objective(trial):
    global best_overall_model, best_overall_loss, best_config
    config = {
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "num_layers": trial.suggest_categorical("num_layers", [2, 4, 6]),
        "d_ff": trial.suggest_categorical("d_ff", [512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64]),
    }
    train_loader = data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=config["batch_size"])
    model = Transformer(len(vocab), len(vocab), config["d_model"], config["num_heads"], config["num_layers"], config["d_ff"], max_length, config["dropout"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    val_loss = float('inf')
    for _ in range(3):
        train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
    if val_loss < best_overall_loss:
        best_overall_loss = val_loss
        best_overall_model = copy.deepcopy(model.state_dict())
        best_config = config
        torch.save(best_overall_model, "best_model.pth")
    return val_loss

# Run Optuna Study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Save best trial to JSON
result_path = "best_model_summary.json"
with open(result_path, 'w') as f:
    json.dump({
        "Validation Loss": study.best_trial.value,
        "Params": study.best_trial.params
    }, f, indent=4)
print(f"\nBest trial result saved to: {result_path}")
