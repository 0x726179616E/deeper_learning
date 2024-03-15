#!/usr/bin/env python3

import numpy as np
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, d_model, heads, ff_dim=2048, dropout=0.1):
        super(Decoder, self).__init__()
        # masked scaled dot-prodouct attention mechanism
        self.attn = nn.MultiheadAttention(d_model, heads, dropout)

        # feedforward neural network 
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

        # normalization via layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # regularization via dropout 
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, target, memory, memory_mask=None, target_mask=None):
        # first sublayer [layernorm -> attention mechanism -> dropout + residual]
        target2 = self.norm1(target)
        q = k = target2
        target2 = self.attn(query=q, key=k, value=target2, key_padding_mask=memory_mask, attn_mask=target_mask)[0]
        target = target + self.dropout1(target2) 

        # second sublayer [layernorm -> feedforward -> dropout + residual]
        target2 = self.norm2(target)
        target2 = self.ff(target2)
        target = target + self.dropout2(target2) 

        return target

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, heads, ff_dim, max_len, vocab_size, dropout=0.1):
        super(Transformer, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([Decoder(d_model, heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, source, source_key_padding_mask=None, source_mask=None):
        source = self.word_emb(source)
        source = self.pos_encoder(source)
        output = source

        for layer in self.layers:
            output = layer(output, memory=source, memory_mask=source_key_padding_mask, target_mask=None)

        output = self.norm(output)
        output = self.fc_out(output)

        return output

if __name__ == "__main__":
    # detect pytorch device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    # define hyperparameters: 
    num_layers = 6    # number of decoder layers
    d_model = 512     # dimensionality of the model
    heads = 8         # number of attention heads
    ff_dim = 2048     # dimensionality of feedforward network
    max_len = 4096    # maximum sequence length
    vocab_size = 1000 # size of the vocabulary
    dropout = 0.1     # dropout rate
    batch_size = 32   # batch size
    seq_len = 50      # sequence length

    # instantiate the model
    model = Transformer(num_layers=num_layers, d_model=d_model, heads=heads, ff_dim=ff_dim, max_len=max_len, vocab_size=vocab_size, dropout=dropout).to(device)

    # sample input sequence
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # optional: create a source mask
    src_mask = None # (batch_size, 1, seq_len)

    # optional: create a key padding mask
    src_key_padding_mask = torch.randint(0, 2, (batch_size, seq_len)).bool().to(device).T
    # src_key_padding_mask = src_key_padding_mask.T

    # forward pass
    output = model(source=src, source_key_padding_mask=src_key_padding_mask, source_mask=src_mask)
    print(f"batch_size: {batch_size}\nsequence length: {seq_len}\nvocabulary size: {vocab_size}\n")
    print(f"output shape: {output.shape}\n") # expected: (batch_size, sequence_length, vocab_size)

    print("COMPLETE: decoder-only transformer")
