#!/usr/bin/env python3

import torch
import torch.nn as nn

# detect torch device
if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# decoder block within transformer as seen in Vaswani et al. (2017)
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, d_ff=2048, dropout=0.1):
        super(DecoderBlock, self).__init__()
        # multihead attention mechanism
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)

        # point-wise feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim),
        )

        # regularization (dropout)
        self.dropout = nn.Dropout(dropout)

        # normalization (layernorm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)


    def forward(self, x, attn_mask=None):
        # first sublayer (multihead attention)
        residual = x
        x_attn = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.dropout(x_attn[0]) + residual
        x = self.norm1(x)

        # second sublayer (feed forward)
        residual = x
        x = self.ff(x)
        x = self.dropout(x) + residual
        x = self.norm2(x)

        return x

# TODO: implement function that can generate masking to preserve transformer's auto-regressive properties

# driver function
def main():
    bs = 32 # batch size
    seq_len = 1024 # max sequence length
    d_model = 512 # dimensionality of model (aka embedding size)
    heads = 8 # number of heads in multihead attention mechanism

    # print important dimensions
    print(f'batch size: {bs}')
    print(f'sequence length: {seq_len}')
    print(f'model dimensionality: {d_model}')
    print(f'number of heads: {heads}')
    print()

    # instantiate decoder block
    decoder = DecoderBlock(d_model, heads)

    # sample input
    x = torch.rand(bs, seq_len, d_model) # [ batch_size, seq_len, embed_dim)
    print(f'input: {x.shape}')
    print()

    # TODO: mask = generate_mask(seq_len) # (seq_len, seq_len)


    # compute output of decoder-only attention block
    out = decoder(x)
    print(f'out: {out.shape}')
    print()

    print("COMPLETE: decoder-only transformer block")

if __name__ == '__main__':
    main()
