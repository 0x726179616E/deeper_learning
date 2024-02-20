#!/usr/bin/env python3

import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_printoptions(sci_mode=False)

# detect torch device
if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# define convnet 
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # convolutional layers
        pass
    def forward(self, x: torch.Tensor):
        # feature maps' dimensions: channels @ width x height
        pass

# load mnist dataset from ubyte file into np array
def load(file_path):
    abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    try: 
        with open(abs_file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    except IOError: 
        print(f'error reading file: {file_path}')
    return data


# driver function
def main():
    # load training set and test set into numpy ararys
    X_train = load('../data/mnist/train-images-idx3-ubyte')[0x10:].reshape((-1,28,28))
    Y_train = load('../data/mnist/train-labels-idx1-ubyte')[8:]
    X_test = load('../data/mnist/t10k-images-idx3-ubyte')[0x10:].reshape((-1,28,28))
    Y_test = load('../data/mnist/t10k-labels-idx1-ubyte')[8:]

    print(f'\ntraining model on {device}...\n')
    print('COMPLETE: convolutional neural network')

# run the entire program
if __name__ == '__main__':
    main()
