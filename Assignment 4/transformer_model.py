import torch
import torch.nn as nn
import ast
import math
import pandas as pd
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

class Vocab:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []
        for t in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            self.add(t)

    def add(self, tok):
        if tok not in self.token2idx:
            self.idx2token.append(tok)
            self.token2idx[tok] = len(self.idx2token) - 1

    def build_from_dataset(self, samples):
        for inp, out in samples:
            for t in inp: self.add(t)
            for t in out: self.add(t)

class MazeDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.items = []
        for _, row in df.iterrows():
            src = ast.literal_eval(row["input_sequence"])
            tgt = ast.literal_eval(row["output_path"])
            self.items.append((row["id"], src, tgt))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(vocab_size, 128, padding_idx=pad_idx)
        self.pos = PositionalEncoding(128)

        enc = nn.TransformerEncoderLayer(128, 8, 512, dropout=0.1, norm_first=True)
        dec = nn.TransformerDecoderLayer(128, 8, 512, dropout=0.1, norm_first=True)

        self.encoder = nn.TransformerEncoder(enc, 6)
        self.decoder = nn.TransformerDecoder(dec, 6)

        self.fc = nn.Linear(128, vocab_size)
        self.fc.weight = self.embed.weight  

    def forward(self, src, tgt):
        src_emb = self.pos(self.embed(src))
        tgt_emb = self.pos(self.embed(tgt))

        src_mask = (src == self.pad_idx)
        tgt_mask_pad = (tgt == self.pad_idx)

        T = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(src.device)

        mem = self.encoder(src_emb.transpose(0,1), src_key_padding_mask=src_mask)
        out = self.decoder(
            tgt_emb.transpose(0,1),
            mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_mask_pad,
            memory_key_padding_mask=src_mask
        )

        return self.fc(out.transpose(0,1))

def greedy_decode(model, vocab, src_tokens, max_len=60, device="cpu"):
    model.eval()
    src_idx = torch.tensor([[vocab.token2idx[t] for t in src_tokens]]).to(device)

    tgt = torch.tensor([[vocab.token2idx[SOS_TOKEN]]]).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src_idx, tgt)
            next_token = logits[0,-1].argmax().item()
            tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)
            if vocab.idx2token[next_token] == EOS_TOKEN:
                break

    return [vocab.idx2token[i] for i in tgt[0].tolist()][1:-1]  
