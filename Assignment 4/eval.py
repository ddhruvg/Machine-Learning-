import sys
import ast
import torch
import torch.nn as nn
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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

def tf_greedy_decode(model, vocab, src_tokens, max_len=60, device="cpu"):
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

class Dataset_maze(Dataset):
    
    def __init__(self, df, token_to_id):
        super().__init__()
        self.df = df
        self.inputs = df["input_sequence"].apply(ast.literal_eval).tolist()
        self.token_to_id = token_to_id

    def encode(self, seq):
        return [self.token_to_id[t] for t in seq]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src_tokens = self.inputs[idx]
        src_ids = self.encode(src_tokens)
        return torch.tensor(src_ids, dtype=torch.long)


def collate_rnn(pad_id):
    def src_pd(batch, pad_id_inner):
        return pad_sequence(batch, batch_first=True, padding_value=pad_id_inner)

    def collate(batch):
        src_lengths = torch.tensor([len(s) for s in batch], dtype=torch.long)
        src_padded = src_pd(batch, pad_id)
        return src_padded, src_lengths

    return collate


class Attention(nn.Module):
    def __init__(self, hidden_dim, enc_out_dim, attn_dim=None):
        super().__init__()
        if attn_dim is None:
            attn_dim = hidden_dim
        self.get_a(hidden_dim, enc_out_dim, attn_dim)

    def get_a(self, hidden_dim, enc_out_dim, attn_dim):
        self.W_vect_a = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_vec_a  = nn.Linear(attn_dim, 1,         bias=False)
        self.U_vec_a  = nn.Linear(enc_out_dim, attn_dim, bias=False)

    def forward(self, decoder_state, encoder_outputs, src_mask):
        d = decoder_state.view(decoder_state.size(0), 1, -1)
        k_dec = self.W_vect_a(d)                
        k_enc = self.U_vec_a(encoder_outputs)    
        z = k_dec + k_enc                        
        att_hidden = torch.tanh(z)               
        scores = self.v_vec_a(att_hidden).squeeze(-1) 
        scores = scores + (src_mask.logical_not() * -1e9)
        weights = torch.softmax(scores, dim=-1)      
        ctx = torch.matmul(weights.unsqueeze(1), encoder_outputs).squeeze(1)  
        return ctx, weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            dropout=0.2,
        )

    def forward(self, src, src_lengths):
        emb = self.embedding(src)        
        emb_t = emb.transpose(0, 1)        
        lengths_cpu = src_lengths.detach().cpu()
        sorted_len, sort_idx = torch.sort(lengths_cpu, descending=True)
        emb_sorted = emb_t[:, sort_idx, :]  
        packed = pack_padded_sequence(
            emb_sorted,
            lengths=sorted_len,
            batch_first=False,
            enforce_sorted=True,
        )
        packed_out, hidden_sorted = self.rnn(packed)
        out_sorted, _ = pad_packed_sequence(packed_out, batch_first=False)
        _, inv_idx = torch.sort(sort_idx)
        out    = out_sorted[:, inv_idx, :]       
        hidden = hidden_sorted[:, inv_idx, :]    
        out = out.transpose(0, 1)                
        return out, hidden



def greedy_decode_rnn(model, src, src_lengths, path_start_id, path_end_id, max_len=50):\

    model.eval()
    device = model.device

    src = src.to(device)
    src_lengths = src_lengths.to(device)

    with torch.no_grad():
        encoder_outputs, enc_hidden, src_mask = model.encode(src, src_lengths)

        batch_size = src.size(0)
        cur_token = torch.full(
            (batch_size,), path_start_id, dtype=torch.long, device=device
        )
        hidden = enc_hidden

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        preds = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            logits, hidden, _ = model.decoder.forward_step(
                cur_token, hidden, encoder_outputs, src_mask
            )
            next_token = logits.argmax(-1)  

            for i in range(batch_size):
                if not finished[i]:
                    preds[i].append(int(next_token[i].item()))
                    if next_token[i].item() == path_end_id:
                        finished[i] = True

            if finished.all():
                break

            cur_token = next_token

    return preds



class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, enc_out_dim, pad_id):
        super().__init__()
        self.attention = Attention(hidden_dim, enc_out_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.RNN(
            input_size=emb_dim + enc_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc_out = nn.Linear(hidden_dim + enc_out_dim + emb_dim, vocab_size)
        
        

    def forward_step(self, input_token, last_hidden, encoder_outputs, src_mask):
        emb_now = self.embedding(input_token)          
        top_state = last_hidden[-1]               
        ctx_vec, attn_wts = self.attention(top_state, encoder_outputs, src_mask)
        step_inp = torch.cat([emb_now, ctx_vec], dim=-1).unsqueeze(1)  
        step_out, new_hidden = self.rnn(step_inp, last_hidden)
        step_out = step_out[:, 0, :]                 
        combo_vec = torch.cat([step_out, ctx_vec, emb_now], dim=-1)    
        logits = self.fc_out(combo_vec)               
        return logits, new_hidden, attn_wts




class Sequence2Sequence(nn.Module):
    def __init__(self, encoder, decoder, pad_id, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.device = device

    def create_src_mask(self, src):
        return (src != self.pad_id).long()

    def encode(self, src, src_lengths):
        src = src.to(self.device)
        src_lengths = src_lengths.to(self.device)
        enc_outputs, enc_hidden = self.encoder(src, src_lengths)
        src_mask = self.create_src_mask(src)
        return enc_outputs, enc_hidden, src_mask

#Transformer 

def build_vocab_from_token_to_id(token_to_id):
    vocab = Vocab()
    vocab.token2idx = token_to_id
    max_idx = max(token_to_id.values())
    idx2token = [None] * (max_idx + 1)
    for tok, idx in token_to_id.items():
        idx2token[idx] = tok
    vocab.idx2token = idx2token
    return vocab

def remap_old_attention_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if "decoder.attention.W_a" in k:
            k = k.replace("W_a", "W_vect_a")
        elif "decoder.attention.U_a" in k:
            k = k.replace("U_a", "U_vec_a")
        elif "decoder.attention.v_a" in k:
            k = k.replace("v_a", "v_vec_a")
        new_state[k] = v
    return new_state



if __name__ == "__main__":
    
    model_path  = sys.argv[1]
    model_type  = sys.argv[2].lower()   # "rnn" or "transformer"
    data_path   = sys.argv[3]
    output_path = sys.argv[4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(data_path)

    
    #  CASE 1: TRANSFORMER
   
    if model_type == "transformer":
        

        ckpt = torch.load(model_path, map_location=device)

     
        token_to_id = ckpt["token_to_id"]
        vocab = build_vocab_from_token_to_id(token_to_id)
        pad_idx = vocab.token2idx["<PAD>"]
        vocab_size = len(vocab.idx2token)
        print(f"Vocab size from checkpoint: {vocab_size}")

        model = TransformerModel(vocab_size, pad_idx).to(device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
        print("Transformer model weights loaded.\n")

        outputs = []
        for _, row in df.iterrows():
            src_tokens = ast.literal_eval(row["input_sequence"])
            pred_tokens = tf_greedy_decode(model, vocab, src_tokens, device=device)
            outputs.append(" ".join(pred_tokens))

        out_df = pd.DataFrame({
            "id": df["id"],
            "input_sequence": df["input_sequence"],
            "maze_type": df["maze_type"],
            "output_path": outputs,
        })
        out_df.to_csv(output_path, index=False)
        

   
    #  CASE 2: RNN
    
    elif model_type == "rnn":
       

        ckpt = torch.load(model_path, map_location=device)

        token_to_id   = ckpt["token_to_id"]
        pad_id        = ckpt["pad_id"]
        path_start_id = ckpt["path_start_id"]
        path_end_id   = ckpt["path_end_id"]

        vocab_size = len(token_to_id)
        id_to_token = {v: k for k, v in token_to_id.items()}

        
        embedding_dim = 128
        hidden_dim    = 512
        num_layers    = 2
        enc_out_dim   = hidden_dim

        encoder = Encoder(
            vocab_size=vocab_size,
            emb_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pad_id=pad_id,
        )
        decoder = Decoder(
            vocab_size=vocab_size,
            emb_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            enc_out_dim=enc_out_dim,
            pad_id=pad_id,
        )
        model = Sequence2Sequence(encoder, decoder, pad_id, device).to(device)

        
        fixed_state_dict = remap_old_attention_keys(ckpt["model_state_dict"])

        model.load_state_dict(fixed_state_dict, strict=True)
        model.eval()
        

        dataset = Dataset_maze(df, token_to_id=token_to_id)
        collate_fn = collate_rnn(pad_id)
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
        )

        all_pred_token_lists = []
        for src_batch, src_lengths in loader:
            batch_pred_ids = greedy_decode_rnn(
                model,
                src_batch,
                src_lengths,
                path_start_id=path_start_id,
                path_end_id=path_end_id,
                max_len=50,
            )
            for seq_ids in batch_pred_ids:
                tokens = []
                for tid in seq_ids:
                    tok = id_to_token[tid]
                    tokens.append(tok)
                    if tok == "<PATH_END>":
                        break
                all_pred_token_lists.append(str(tokens))

        assert len(all_pred_token_lists) == len(df)

        out_df = df.copy()
        out_df["output_path"] = all_pred_token_lists
        out_df.to_csv(output_path, index=False)
        


