#!/usr/bin/env python3

import math
import numpy as np

# rectified linear unit 
def relu(z):
    return np.maximum(0, z)

# gaussian error linear unit (as specified in: https://arxiv.org/abs/1606.08415)
def gelu(z):
    return 0.5 * z * (1 + np.tanh(math.sqrt(2/np.pi) * (z + 0.044715 * np.power(x,3))))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# scaled dot-product attention
def attention(queries: np.ndarray, keys: np.ndarray, values: np.ndarray):
    d = len(keys)
    return softmax((queries @ keys.T) / math.sqrt(d)) @ values

# multihead attention


if __name__ == "__main__":
    print("OK")
