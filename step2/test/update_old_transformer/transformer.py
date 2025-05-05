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

# ========== RailFence Transformer ==========

class RailFenceTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3, d_ff=256, max_length=300, dropout=0.1, max_rail=5):
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)

        self.rail_embedding = nn.Embedding(max_rail, d_model)
        self.rail_position = nn.Parameter(torch.randn(max_rail, d_model))

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
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
            dec_output = layer(dec_output, enc_output,
                             tgt_mask=causal_mask,
                             memory_key_padding_mask=src_padding_mask,
                             tgt_key_padding_mask=tgt_padding_mask)

        return self.fc(dec_output.transpose(0, 1))

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
    # Load the best configuration
    with open("best_model_summary.json") as f:
        best_config = json.load(f)["Params"]

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv('train_augmented.csv')
    inputs, outputs = df['Input'].tolist(), df['Output'].tolist()
    train_in, val_in, train_out, val_out = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Prepare data
    max_length = 512
    vocab = Vocabulary()
    train_dataset = CipherDataset(train_in, train_out, vocab, max_length)
    val_dataset = CipherDataset(val_in, val_out, vocab, max_length)
    train_loader = data.DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=best_config["batch_size"])

    # Model setup with the best parameters
    model = RailFenceTransformer(
        vocab_size=len(vocab),
        d_model=best_config["d_model"],
        num_heads=best_config["num_heads"],
        num_layers=best_config["num_layers"],
        d_ff=best_config["d_ff"],
        max_length=max_length,
        dropout=best_config["dropout"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=best_config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token)

    # Train the model
    model = train_model(model, train_loader, val_loader, optimizer, criterion, device, vocab, epochs=20)

    # Load the best trained model
    model.load_state_dict(torch.load('best_parameter.pth'))



    # File paths
    input_file = 'test_augmented.csv'
    output_file = 'test_results.csv'

    print("Starting decryption...")

    # Load CSV into DataFrame
    df = pd.read_csv(input_file)

    # Ensure there are at least 3 columns
    while df.shape[1] < 3:
        df[f'col_{df.shape[1]}'] = ''

    # Decrypt text in the first column, store in the third
    def decrypt_if_valid(text):
        if pd.notnull(text):
            return decrypt_text(model, str(text), vocab, max_length, device)
        return ''

    df.iloc[:, 2] = df.iloc[:, 0].apply(decrypt_if_valid)

    # Save results to new CSV
    df.to_csv(output_file, index=False)

    print(f"Decryption completed and saved to {output_file}")
    print("All done!")

