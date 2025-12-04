import argparse
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt 



class Datset_of_Maze(Dataset):
   
    def __init__(
        self,
        csv_path,
        token_to_id = None,
        path_start_token = "<START_PATH>",
        path_end_token = "<END_PATH>",
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)

        self.define_input_output()

        self.get_pathtokens(path_end_token,path_start_token)

        self.token_to_id = self.build_vocab(self.inputs, self.outputs) if token_to_id is None else token_to_id    
        self._init_vocab__()

    def get_pathtokens(self,path_end_token,path_start_token):
        self.path_end_token = path_end_token
        self.path_start_token = path_start_token


    def define_input_output(self): 
        def get_in_and_outs():
            self.inputs = self.df["input_sequence"].apply(eval).tolist()
            self.outputs = self.df["output_path"].apply(eval).tolist()   
        get_in_and_outs()    

    def _init_vocab__(self): 
        def set_ids():
            self.path_start_id = self.token_to_id[self.path_start_token]
            self.path_end_id = self.token_to_id[self.path_end_token]

        set_ids()    
        self.pad_id = self.token_to_id["<padding>"]
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def build_vocab(self, inputs, outputs):
        vocab = set()
        for seq in inputs:
            vocab.update(seq)
        for seq in outputs:
            vocab.update(seq)

   
        vocab.add("<padding>")

        token_to_id = {tok: i for i, tok in enumerate(sorted(vocab))}
        return token_to_id

    def encoded(self, seq: List[str]) -> List[int]:
        return [self.token_to_id[t] for t in seq]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        src_tokens = self.inputs[idx]
        src_ids = self.encoded(src_tokens)

        
        y_tokens = self.outputs[idx]
        y_ids = self.encoded(y_tokens)

        dec_in_ids = [self.path_start_id] + y_ids[:-1] if len(y_ids) > 0 else [self.path_start_id]
        dec_out_ids = y_ids  

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(dec_in_ids, dtype=torch.long),
            torch.tensor(dec_out_ids, dtype=torch.long),
        )


def collate_function(pad_id: int):
    def collate(batch):
        src_seqs, tgt_in_seqs, tgt_out_seqs = zip(*batch)

        src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
        tgt_lengths = torch.tensor([len(s) for s in tgt_in_seqs], dtype=torch.long)

        src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=pad_id)
        tgt_in_padded = pad_sequence(tgt_in_seqs, batch_first=True, padding_value=pad_id)
        tgt_out_padded = pad_sequence(tgt_out_seqs, batch_first=True, padding_value=pad_id)

        return src_padded, src_lengths, tgt_in_padded, tgt_out_padded, tgt_lengths

    return collate



class Atention(nn.Module):
    """
    Bahdanau attention mechanism.
    Eij = v_a^T tanh(W_a s_{i-1} + U_a h_j) Eij: score for decoder step i and encoder step j
    alpha_ij = softmax_j(Eij) (attention weights)
    context_
    this is from the equation 3 in the paper:
    c_i = sum_j alpha_ij h_j"""
   

    def __init__(self, hidden_dim: int, enc_out_dim: int, attn_dim: int = None):
        super().__init__()
        if attn_dim is None:
            attn_dim = hidden_dim

        
        
        self.V_model_a = nn.Linear(attn_dim, 1, bias=False)

        self.W_model_a = nn.Linear(hidden_dim, attn_dim, bias=False)

        self.U_model_a = nn.Linear(enc_out_dim, attn_dim, bias=False)

    def forward(self, decoder_state, encoder_outputs, src_mask):
       
       
        dec = decoder_state.unsqueeze(1)

        
        Eij = self.V_model_a(torch.tanh(self.W_model_a(dec) + self.U_model_a(encoder_outputs))).squeeze(-1)
      


        Eij = Eij.masked_fill(src_mask == 0, -1e9-9)

        alpha_ij = torch.softmax(Eij, dim=-1)        
        context = torch.bmm(alpha_ij.unsqueeze(1),    
                            encoder_outputs).squeeze(1)  

        return context, alpha_ij


class Initializer:

    def __init__(self, module):
        self.module = module
        pass

    def init_weights(self):
       
        if isinstance(self.module, nn.Embedding):
            nn.init.xavier_uniform_(self.module.weight.data)
        elif isinstance(self.module, nn.Linear):
            nn.init.xavier_uniform_(self.module.weight.data)
            if self.module.bias is not None:
                nn.init.zeros_(self.module.bias.data)
        else:
            pass       
        
        self.get_rnn_init_weights(self.module)
    def get_rnn_init_weights(self, module):
        if isinstance(module, nn.RNN):
            for nme, par in module.named_parameters():
                if "weight_ih" in nme:
                    nn.init.xavier_uniform_(par.data)
                elif "weight_hh" in nme:
                    nn.init.orthogonal_(par.data)
                elif "bias" in nme:
                    nn.init.zeros_(par.data)



class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,   
        )

    def forward(self, src, src_lengths):
        
        embedded = self.embedding(src)  
        packed = pack_padded_sequence(
            embedded,
            lengths=src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.rnn(packed)

   
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1),
        )
        

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, enc_out_dim, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=emb_dim + enc_out_dim,  
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2, 
        )

        self.attention = Atention(hidden_dim, enc_out_dim)
        self.fc_out = nn.Linear(hidden_dim + enc_out_dim + emb_dim, vocab_size)

    def forward_step(self, input_token, last_hidden, encoder_outputs, src_mask):
        embedded = self.embedding(input_token).unsqueeze(1)  
        dec_state = last_hidden[-1]                         

        context, attn_weights = self.attention(dec_state, encoder_outputs, src_mask)
        context = context.unsqueeze(1)                       

        rnn_input = torch.cat([embedded, context], dim=-1)  
        rnn_output, hidden = self.rnn(rnn_input, last_hidden)
        rnn_output = rnn_output.squeeze(1)                  

        context = context.squeeze(1)                        
        embedded = embedded.squeeze(1)                    

        logits = self.fc_out(torch.cat([rnn_output, context, embedded], dim=-1))
        return logits, hidden, attn_weights

    def forward(self, tgt_in, encoder_outputs, src_mask, hidden, teacher_forcing_ratio=0.5):
        """
        tgt_in: (batch, tgt_len)   [<PATH_START>, y0, ..., y_{T-1}]
        """
        batch_size, tgt_len = tgt_in.shape
        vocab_size = self.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=tgt_in.device)

        input_token = tgt_in[:, 0]  
        for t in range(tgt_len):
            logits, hidden, _ = self.forward_step(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = logits

            if t + 1 < tgt_len:  
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = logits.argmax(-1)
                input_token = tgt_in[:, t + 1] if teacher_force else top1

        return outputs


class Sequence2Sequence(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_id: int, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.device = device

    def create_src_mask(self, src):
        
        return (src != self.pad_id).long()

    def forward(self, src, src_lengths, tgt_in, teacher_forcing_ratio=0.5):
        src = src.to(self.device)
        src_lengths = src_lengths.to(self.device)
        tgt_in = tgt_in.to(self.device)

        encoder_outputs, enc_hidden = self.encoder(src, src_lengths)
        src_mask = self.create_src_mask(src)

        outputs = self.decoder(
            tgt_in,
            encoder_outputs,
            src_mask,
            enc_hidden,  
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return outputs

    def greedy_decode(self, src, src_length, max_len, path_start_id, path_end_id):
       
        self.eval()
        with torch.no_grad():
            src_length = src_length.to(self.device)
            true_len = int(src_length[0].item())
            src = src[:, :true_len].to(self.device)

            encoder_outputs, enc_hidden = self.encoder(src, src_length)
            src_mask = self.create_src_mask(src)

            input_token = torch.full(
                (1,), path_start_id, dtype=torch.long, device=self.device
            )
            hidden = enc_hidden

            pred_tokens = []
            for _ in range(max_len):
                logits, hidden, _ = self.decoder.forward_step(
                    input_token, hidden, encoder_outputs, src_mask
                )
                next_token = logits.argmax(-1)  # (1,)
                tok_id = next_token.item()
                pred_tokens.append(tok_id)
                if tok_id == path_end_id:
                    break
                input_token = next_token

        return pred_tokens



def train_one_epoch(
    model, dataloader, optimizer, criterion, device, teacher_forcing_ratio, epoch, total_epochs,
    print_every=50,
):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (src, src_lengths, tgt_in, tgt_out, tgt_lengths) in enumerate(dataloader, start=1):
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        optimizer.zero_grad()
        outputs = model(src, src_lengths, tgt_in, teacher_forcing_ratio)
        batch_size, tgt_len, vocab_size = outputs.shape

        outputs_flat = outputs.reshape(-1, vocab_size)
        tgt_out_flat = tgt_out.reshape(-1)

        loss = criterion(outputs_flat, tgt_out_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if (batch_idx % print_every == 0) or (batch_idx == len(dataloader)):
            print(
                f"[Epoch {epoch}/{total_epochs}] "
                f"Batch {batch_idx}/{len(dataloader)} "
                f"Loss: {loss.item():.4f}"
            )

    return epoch_loss / len(dataloader)


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, src_lengths, tgt_in, tgt_out, tgt_lengths in dataloader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            outputs = model(src, src_lengths, tgt_in, teacher_forcing_ratio=0.0)
            batch_size, tgt_len, vocab_size = outputs.shape

            outputs_flat = outputs.reshape(-1, vocab_size)
            tgt_out_flat = tgt_out.reshape(-1)

            loss = criterion(outputs_flat, tgt_out_flat)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def strip_seq(ids, pad_id, end_id):
   
    out = []
    for t in ids:
        if t == pad_id:
            break
        out.append(int(t))
        if t == end_id:
            break
    return out


def token_f1(pred, true):
    
    L = min(len(pred), len(true))
    if L == 0:
        return 0.0

    tp = sum(1 for i in range(L) if pred[i] == true[i])
    fp = len(pred) - tp
    fn = len(true) - tp

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_metrics(model, dataloader, pad_id, path_end_id, device):
    
    model.eval()
    total = 0
    exact_match = 0
    f1_sum = 0.0

    with torch.no_grad():
        for src, src_lengths, tgt_in, tgt_out, tgt_lengths in dataloader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            outputs = model(src, src_lengths, tgt_in, teacher_forcing_ratio=0.0)
            pred_ids_batch = outputs.argmax(dim=-1).cpu().tolist()
            true_ids_batch = tgt_out.cpu().tolist()

            batch_size = len(pred_ids_batch)
            for i in range(batch_size):
                pred_ids = pred_ids_batch[i]
                true_ids = true_ids_batch[i]

                pred_seq = strip_seq(pred_ids, pad_id, path_end_id)
                true_seq = strip_seq(true_ids, pad_id, path_end_id)

                if pred_seq == true_seq:
                    exact_match += 1
                f1_sum += token_f1(pred_seq, true_seq)
                total += 1

    seq_acc = exact_match / total if total > 0 else 0.0
    avg_f1 = f1_sum / total if total > 0 else 0.0
    return seq_acc, avg_f1




def main():
  
    default_train = "/kaggle/input/dhruvg/data/train_6x6_mazes.csv"
    default_val   = "/kaggle/input/dhruvg/data/val_6x6_mazes.csv"
    default_test  = "/kaggle/input/dhruvg/data/test_6x6_mazes.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default=default_train,
                        help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, default=default_val,
                        help="Path to validation CSV")
    parser.add_argument("--test_csv", type=str, default=default_test,
                        help="Path to test CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--print_every", type=int, default=50)





    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    train_dataset = Datset_of_Maze(args.train_csv)
    token_to_id = train_dataset.token_to_id
    pad_id = train_dataset.pad_id
    path_start_id = train_dataset.path_start_id
    path_end_id = train_dataset.path_end_id

    val_dataset = Datset_of_Maze(args.val_csv, token_to_id=token_to_id)
    test_dataset = Datset_of_Maze(args.test_csv, token_to_id=token_to_id)

    collate_fn = collate_function(pad_id)

    
    full_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
   
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

 
    easy_train_loader = None
    if "maze_type" in train_dataset.df.columns:
        easy_indices = [
            i for i, t in enumerate(train_dataset.df["maze_type"])
            if "forkless" in str(t).lower()
        ]
        if len(easy_indices) > 0:
            easy_subset = Subset(train_dataset, easy_indices)
            easy_train_loader = DataLoader(
                easy_subset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            print(f"Curriculum: {len(easy_indices)} forkless mazes for warm-up training.")
        else:
            print("Curriculum: no forkless mazes found; training on full set only.")
    else:
        print("Curriculum: 'maze_type' column not found; training on full set only.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    vocab_size = len(token_to_id)
    enc_out_dim = args.hidden_dim  # regular RNN encoder

    encoder = Encoder(
        vocab_size=vocab_size,
        emb_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_id=pad_id,
    )
    decoder = Decoder(
        vocab_size=vocab_size,
        emb_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        enc_out_dim=enc_out_dim,
        pad_id=pad_id,
    )

    
    init_weights = Initializer.init_weights
    encoder.apply(init_weights)
    decoder.apply(init_weights)

    model = Sequence2Sequence(encoder, decoder, pad_id, device).to(device)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

   

    
    train_losses, val_losses = [], []
    train_seq_accs, val_seq_accs, test_seq_accs = [], [], []
    train_f1s, val_f1s, test_f1s = [], [], []

    
    curriculum_epochs = 8 if easy_train_loader is not None else 0

    for epoch in range(1, args.epochs + 1):
        if easy_train_loader is not None and epoch <= curriculum_epochs:
            train_loader = easy_train_loader
            
        else:
            train_loader = full_train_loader
   
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            epoch=epoch,
            total_epochs=args.epochs,
            print_every=args.print_every,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)



    save_model = model.module if isinstance(model, nn.DataParallel) else model

    torch.save(
        {
            "model_state_dict": save_model.state_dict(),
            "token_to_id": token_to_id,
            "pad_id": pad_id,
            "path_start_id": path_start_id,
            "path_end_id": path_end_id,
            
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_seq_accs": train_seq_accs,
            "val_seq_accs": val_seq_accs,
            "test_seq_accs": test_seq_accs,
            "train_f1s": train_f1s,
            "val_f1s": val_f1s,
            "test_f1s": test_f1s,
        },
        "rnn_weights.pt",
    )

main()

