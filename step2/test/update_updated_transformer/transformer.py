import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import string
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook

# Set random seed
torch.manual_seed(42)

# ========== Vocabulary ==========
class Vocabulary:
    def __init__(self):
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
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices if idx not in {self.pad_token, self.sos_token, self.eos_token}])

# ========== Dataset ==========
class CipherDataset(data.Dataset):
    def __init__(self, inputs, outputs, vocab, max_length):
        self.inputs = inputs
        self.outputs = outputs
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_encoded = [self.vocab.sos_token] + self.vocab.encode(self.inputs[idx]) + [self.vocab.eos_token]
        output_encoded = [self.vocab.sos_token] + self.vocab.encode(self.outputs[idx]) + [self.vocab.eos_token]
        input_padded = input_encoded + [self.vocab.pad_token] * (self.max_length - len(input_encoded))
        output_padded = output_encoded + [self.vocab.pad_token] * (self.max_length - len(output_encoded))
        return torch.tensor(input_padded[:self.max_length]), torch.tensor(output_padded[:self.max_length])

# ========== Transformer Components ==========
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, V)

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
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
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
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
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
        src = self.dropout(self.pos_enc(self.src_emb(src)))
        tgt = self.dropout(self.pos_enc(self.tgt_emb(tgt)))
        for layer in self.enc_layers:
            src = layer(src, src_mask)
        for layer in self.dec_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        return self.fc(tgt)

# ========== Train / Eval ==========
def train_model(model, train_loader, val_loader, optimizer, criterion, device, vocab, epochs=10):
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(src, tgt[:, :-1])
            loss = criterion(out.view(-1, len(vocab)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, device, vocab)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_parameter.pth')
    return model

def evaluate_model(model, loader, criterion, device, vocab):
    model.eval()
    loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt[:, :-1])
            loss += criterion(out.view(-1, len(vocab)), tgt[:, 1:].reshape(-1)).item()
    return loss / len(loader)

# ========== Inference ==========
def decrypt_text(model, text, vocab, max_length, device):
    model.eval()
    with torch.no_grad():
        src = [vocab.sos_token] + vocab.encode(text) + [vocab.eos_token]
        src = src + [vocab.pad_token] * (max_length - len(src))
        src = torch.tensor(src[:max_length]).unsqueeze(0).to(device)
        tgt = torch.tensor([[vocab.sos_token]]).to(device)
        for _ in range(max_length - 1):
            out = model(src, tgt)
            next_token = out.argmax(-1)[:, -1].item()
            if next_token == vocab.eos_token:
                break
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
        return vocab.decode(tgt[0].tolist())

# ========== Main ==========
if __name__ == "__main__":
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load best params
    with open("best_rail_model_params.json") as f:
        best_config = json.load(f)["Params"]

    # Load data
    df = pd.read_csv('train_augmented.csv')
    inputs, outputs = df['Input'].tolist(), df['Output'].tolist()
    train_in, val_in, train_out, val_out = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Prepare data
    max_length = 512
    vocab = Vocabulary()
    train_dataset = CipherDataset(train_in, train_out, vocab, max_length)
    val_dataset = CipherDataset(val_in, val_out, vocab, max_length)
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=64)

    # Model setup
    model = Transformer(len(vocab), len(vocab),
                        256,
                        4,
                        3,
                        1024,
                        max_length,
                        0.19227956996136614).to(device)

    optimizer = optim.Adam(model.parameters(), lr=	0.0003871508280743543)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token)

    # Train
    train_model(model, train_loader, val_loader, optimizer, criterion, device, vocab, epochs=20)

    # Load trained model
    model.load_state_dict(torch.load('best_parameter.pth'))

  
    input_file = 'test_augmented.csv'
    output_file = 'test_results.csv'

    print("Starting decryption...")
    df = pd.read_csv(input_file)

    # Make sure the third column exists (or add it)
    if df.shape[1] < 3:
        for _ in range(3 - df.shape[1]):
            df[f'col_{df.shape[1]}'] = ''

    # Apply decryption on the first column
    def decrypt_if_valid(text):
        if pd.notnull(text):
            return decrypt_text(model, str(text), vocab, max_length, device)
        return ''

    df.iloc[:, 2] = df.iloc[:, 0].apply(decrypt_if_valid)

    # Save to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Decryption completed and saved to {output_file}")
    print("All done!")

