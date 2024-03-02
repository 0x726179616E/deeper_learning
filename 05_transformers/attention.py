#!/usr/bin/env python3

import math
import numpy as np

BS = 32 # batch size
seqlen = 1024 # max length of a sequence
dmodel = 512 # dimensionality of input embeddings
heads = 8 # number of heads 
dk = int(dmodel / heads)
dv = dk

# compute softmax along last dimension of a tensor
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# scaled dot-product attention
def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    d = K.shape[-1]
    scaled_QK = np.dot(Q, K.T) / math.sqrt(d)
    return np.dot(softmax(scaled_QK), V)

# multihead attention 
def multihead_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, h):
    d = Q.shape[-1]
    Wo = np.random.rand(h*d, dmodel)
    heads_list = []
    for i in range(h):
        Wqi = np.random.rand(dmodel, d)
        Wki = np.random.rand(dmodel, d)
        Wvi = np.random.rand(dmodel, d)
        head  = attention()
    np.concatenate()
    



# tensor of input embeddings
X = np.random.rand(BS, seqlen, dmodel)

print(f'X: {X.shape}')
print()

# projection matrices from X to Q,K,V
Wq = np.random.rand(dmodel, dk)
Wk = np.random.rand(dmodel, dk)
Wv = np.random.rand(dmodel, dv)

print(f'Wq: {Wq.shape}')
print(f'Wk: {Wk.shape}')
print(f'Wv: {Wv.shape}')
print()

# compute Q,K,V
Q = np.dot(X, Wq)
K = np.dot(X, Wk)
V = np.dot(X, Wv)

print(f'Q: {Q.shape}')
print(f'K: {K.shape}')
print(f'V: {V.shape}')
print()

print(f'Q: {Q.shape}')
print(f'K: {K.shape}')
print(f'V: {V.shape}')
print()

multihead_attention(Q,K,V, heads)
