#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(sci_mode=False)

# detect torch device
if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# scaled dot-product attention
def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
    dk = torch.tensor(K.shape[-1])
    logits = Q @ K.transpose(-2,-1) / torch.sqrt(dk)

    # handle masking (for decoder)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -9e15)

    attention_weights = F.softmax(logits, dim=-1)
    output = attention_weights @ V
    return output, attention_weights

# multihead attention mechanism
def multihead_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, dmodel: int, mask=None):
    depth = dmodel // num_heads

    # linearly project queries, keys, and values
    Q_proj = torch.randn(*Q.shape, depth)
    K_proj = torch.randn(*K.shape, depth)
    V_proj = torch.randn(*V.shape, depth)
    print(f'Q_proj: {Q_proj.shape}')

    # split projections to fit into each head
    Q_split = torch.split(Q_proj, num_heads, dim=-1)
    K_split = torch.split(K_proj, num_heads, dim=-1)
    V_split = torch.split(V_proj, num_heads, dim=-1)

    outs = [] # list of outputs from each head
    for i in range(num_heads):
        output, _ = attention(Q_split[i], K_split[i], V_split[i], mask)
        outs.append(output)
    
    # concatenate outputs into single tensor
    cat_attn = torch.cat(outs, dim=-1)
    Wo = torch.randn(*cat_attn.shape) 

    # compute final linear projection 
    Y = cat_attn @ Wo.transpose(-2,-1)
    return Y

def main():
    print(f'using device: {device}')
    print()

    heads = 8
    dmodel = 512
    seq_len = 100

    # diagnostic prints
    print(f'heads: {heads}')
    print(f'dmodel: {dmodel}')
    print(f'heads: {seq_len}')
    print()

    # randonly init queries, keys, and values from normal distribution
    Q = torch.randn(seq_len, dmodel)
    K = torch.randn(seq_len, dmodel)
    V = torch.randn(seq_len, dmodel)

    # diagnostic prints 
    print(f'Q: {Q.shape}')
    print(f'K: {K.shape}')
    print(f'V: {V.shape}')
    print()

    # compute multihead attention mechanism
    Y = multihead_attention(Q, K, V, heads, dmodel)

    # diagnostic print
    print(f'Y: {Y.shape}')

    print("\nCOMPLETE: multihead attention mechanism\n")

if __name__ == "__main__":
    main()