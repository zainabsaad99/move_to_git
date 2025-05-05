import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
import string

# Data Preparation

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
        # Special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        # All printable characters
        all_chars = list(string.printable)

        # Create vocabulary
        self.char2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.char2idx.update({char: idx+len(special_tokens) for idx, char in enumerate(all_chars)})

        # Reverse mapping
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def __len__(self):
        return len(self.char2idx)

    def encode(self, text):
        return [self.char2idx.get(char, self.unk_token) for char in text]

    def decode(self, indices):
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices if idx not in {self.pad_token, self.sos_token, self.eos_token}])

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
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]

        # Encode with SOS and EOS tokens
        input_encoded = [self.vocab.sos_token] + self.vocab.encode(input_text) + [self.vocab.eos_token]
        output_encoded = [self.vocab.sos_token] + self.vocab.encode(output_text) + [self.vocab.eos_token]

        # Pad sequences
        input_padded = input_encoded + [self.vocab.pad_token] * (self.max_length - len(input_encoded))
        output_padded = output_encoded + [self.vocab.pad_token] * (self.max_length - len(output_encoded))

        # Truncate if necessary
        input_padded = input_padded[:self.max_length]
        output_padded = output_padded[:self.max_length]

        return torch.tensor(input_padded), torch.tensor(output_padded)

# Transformer Model (same as your implementation)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shapes: (batch_size, num_heads, seq_len, d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # Source mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)

        # Target mask
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask  # (batch_size, 1, tgt_len, tgt_len)

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.train()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])

            loss = criterion(output.contiguous().view(-1, len(vocab)),
                           tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_parameter.pth')

    return model

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1, len(vocab)),
                           tgt[:, 1:].contiguous().view(-1))
            total_loss += loss.item()

    return total_loss / len(data_loader)

# Inference Function
def decrypt_text(model, text, vocab, max_length, device):
    model.eval()
    with torch.no_grad():
        # Encode the input text
        encoded = [vocab.sos_token] + vocab.encode(text) + [vocab.eos_token]
        encoded = encoded + [vocab.pad_token] * (max_length - len(encoded))
        encoded = torch.tensor(encoded[:max_length]).unsqueeze(0).to(device)

        # Initialize target with SOS token
        target = torch.tensor([[vocab.sos_token]]).to(device)

        for _ in range(max_length - 1):
            output = model(encoded, target)
            next_token = output.argmax(2)[:, -1].item()
            if next_token == vocab.eos_token:
                break
            target = torch.cat([target, torch.tensor([[next_token]]).to(device)], dim=1)

        # Decode the output
        decrypted = vocab.decode(target[0].cpu().numpy())
        return decrypted

# Main Execution
if __name__ == "__main__":
    # Load and prepare data
    inputs, outputs = load_data('train_augmented.csv')

    # Create vocabulary
    vocab = Vocabulary()

    # Split data into train and validation
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
        inputs, outputs, test_size=0.2, random_state=42
    )

    # Create datasets
    max_length = 512  # Adjust based on your data
    train_dataset = CipherDataset(train_inputs, train_outputs, vocab, max_length)
    val_dataset = CipherDataset(val_inputs, val_outputs, vocab, max_length)

    # Create data loaders
    batch_size = 64
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=512,
        max_seq_length=max_length,
        dropout=0.20284372216007185
    ).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token)
    optimizer = optim.Adam(model.parameters(), lr=0.0003156554523354843, betas=(0.9, 0.98), eps=1e-9)

    # Train the model
    num_epochs = 20
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)


