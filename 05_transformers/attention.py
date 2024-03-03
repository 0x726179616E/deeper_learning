#!/usr/bin/env python3

import torch
import torch.nn.functional as F

# detect torch device
if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# scaled dot-product attention
def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
    """
    Y = softmax((Q @ K.T)/sqrt(dk)) @ V
    """
    # get dimensionality of input vectors
    dk = torch.tensor(Q.shape[-1]) 

    # compute logits (attention scores b/w queries and keys)
    logits = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(dk)

    # apply look-ahead masking (in the auto-regressive case)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, float('-inf'))

    # softmax over last dimension of logits (scores)
    attention_weights = F.softmax(logits, dim=-1)

    # apply attention weights to values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# multihead attention mechanism
class multihead_attention(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(multihead_attention, self).__init__()
        self.embed_size = embed_size # aka dmodel in AIAYN paper
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "embedding size must be divisible by the number of heads"

        # linear projections for q, k, v and final attention scores
        self.q_proj = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, q, k, v, mask):
        N = q.shape[0]
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]

        # split embeddings into `self.heads` different pieces
        q = q.reshape(N, q_len, self.heads, self.head_dim)
        k = k.reshape(N, k_len, self.heads, self.head_dim)
        v = v.reshape(N, v_len, self.heads, self.head_dim)

        # linearly project q, k, and v before computing attention scores 
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # apply attention to each head
        attention_scores, _ = attention(q, k, v, mask)

        # concatenate results from each attention head
        attention_scores = attention_scores.permute(0,2,1,3).reshape(N, q_len, self.heads * self.head_dim)
        
        # apply final linear projection to attention scores
        out = self.fc_out(attention_scores)

        return out

# driver function
def main():
    print(f'pytorch is using device: {device}\n')

    N = 32 # batch size
    heads = 8 # number of heads
    seq_len = 100 # sequence length
    embed_size = 512 # aka dmodel (dimensionality of the model)

    # diagnostic prints for "hyperparameters"
    print(f'batch size (N): {N}')
    print(f'number of heads: {heads}')
    print(f'sequence length: {100}')
    print(f'embeddings\'s size (dmodel): {embed_size}')
    print()

    # instantiate model
    model = multihead_attention(embed_size, heads)

    # create random dummy tensors for queries, keys, and values
    Q = torch.randn((N, seq_len, embed_size))
    K = torch.randn((N, seq_len, embed_size))
    V = torch.randn((N, seq_len, embed_size))
    mask = None

    # diagnostic prints for input tensors' shapes
    print(f'Q shape: {Q.shape}')
    print(f'K shape: {K.shape}')
    print(f'V shape: {V.shape}')
    print()

    # compute multihead attention 
    Y = model(Q, K, V, mask)
    print(f'Y shape: {Y.shape}') # expecting (N=1, seq_len=100, embed_size=512)

    print("\nCOMPLETE: multihead attention\n")

if __name__ == "__main__":
    main()