import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import math
import optuna
import json
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Vocabulary
class Vocabulary:
    def __init__(self):
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?')")
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        for i, c in enumerate(chars):
            self.char2idx[c] = i + 3
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, text, max_length):
        encoded = [self.sos_token] + [self.char2idx.get(c, 0) for c in text] + [self.eos_token]
        encoded += [self.pad_token] * (max_length - len(encoded))
        return encoded[:max_length]

    def decode(self, indices):
        return ''.join([self.idx2char.get(idx, '') for idx in indices if idx not in [self.pad_token, self.sos_token, self.eos_token]])

    def __len__(self):
        return len(self.char2idx)

# Dataset
class CipherDataset(Dataset):
    def __init__(self, df, vocab, max_length):
        self.inputs = df['Input'].tolist()
        self.outputs = df['Output'].tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = self.vocab.encode(self.inputs[idx], self.max_length)
        output_seq = self.vocab.encode(self.outputs[idx], self.max_length)
        return torch.tensor(input_seq), torch.tensor(output_seq)

# Transformer layers
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)

    def forward(self, x, src_key_padding_mask):
        return self.layer(x, src_key_padding_mask=src_key_padding_mask)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)

    def forward(self, x, memory, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask):
        return self.layer(x, memory, tgt_mask=tgt_mask,
                          memory_key_padding_mask=memory_key_padding_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask)

class RailFenceTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3, d_ff=256, max_length=300, dropout=0.1, max_rail=5):
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.rail_embedding = nn.Embedding(max_rail, d_model)
        self.rail_position = nn.Parameter(torch.randn(max_rail, d_model))
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def _generate_rail_pattern(self, seq_len, max_rail=5):
        pattern = []
        direction = 1
        rail = 0
        for _ in range(seq_len):
            pattern.append(rail)
            rail += direction
            if rail == 0 or rail == max_rail - 1:
                direction *= -1
        return torch.tensor(pattern, dtype=torch.long)

    def forward(self, src, tgt):
        device = src.device
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)
        causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device) == 1).transpose(0, 1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
        rail_pattern = self._generate_rail_pattern(src_len).to(device)
        rail_emb = self.rail_embedding(rail_pattern).unsqueeze(0).expand(batch_size, -1, -1)
        rail_pos = self.rail_position[rail_pattern].unsqueeze(0).expand(batch_size, -1, -1)
        src_emb = self.dropout(self.encoder_embedding(src)) + rail_emb + rail_pos
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.dropout(self.decoder_embedding(tgt))
        tgt_emb = self.positional_encoding(tgt_emb)
        enc_output = src_emb.transpose(0, 1)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_key_padding_mask=src_padding_mask)
        dec_output = tgt_emb.transpose(0, 1)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask=causal_mask,
                               memory_key_padding_mask=src_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask)
        return self.fc(dec_output.transpose(0, 1))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                total_loss += loss.item()
        print(f"Epoch: {epoch+1}, Validation Loss: {total_loss / len(val_loader):.4f}")
    return model

def objective(trial):
    valid_combinations = [
        {"d_model": 128, "num_heads": 4},
        {"d_model": 128, "num_heads": 8},
        {"d_model": 192, "num_heads": 4},
        {"d_model": 192, "num_heads": 6},
        {"d_model": 256, "num_heads": 4},
        {"d_model": 256, "num_heads": 8}
    ]
    combo = trial.suggest_categorical("d_model_num_heads", valid_combinations)
    d_model = combo["d_model"]
    num_heads = combo["num_heads"]

    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4])
    d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = RailFenceTransformer(
        vocab_size=len(vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_length=max_length,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=False)

    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=10,
        device=device,
        scheduler=scheduler
    )

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    df = pd.read_csv("train_augmented.csv", encoding='utf-8')
    df['input_len'] = df['Input'].str.len()
    df['output_len'] = df['Output'].str.len()
    base_length = int(max(df['input_len'].quantile(0.99), df['output_len'].quantile(0.99))) + 10
    max_length = ((base_length + 31) // 32) * 32
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['Output'].str.len())
    vocab = Vocabulary()
    train_dataset = CipherDataset(train_df, vocab, max_length)
    val_dataset = CipherDataset(val_df, vocab, max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64 if torch.cuda.is_available() else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("\nBest Trial:")
    print(study.best_trial)

    with open("best_rail_model_params.json", "w") as f:
        json.dump({
            "Validation Loss": study.best_trial.value,
            "Params": study.best_trial.params
        }, f, indent=4)
    print("\nBest hyperparameters saved to best_rail_model_params.json")

