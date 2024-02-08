#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.set_printoptions(sci_mode=False)

# detect torch device
if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# TODO: implement architecture from https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
"""
    hyperparams: 
    -> 30 epochs
    -> l2 regularization: 0.005
    -> dropout: 25% after the pooling layers and some fully connected layers
    -> batchnorm: after every set of layers (conv + pool + fc)
    -> learning rate scheduler: decrease by 0.2x after waiting for 2 epochs at a plateau

    architecture:
    -> conv 
    -> relu
    -> conv 
    -> batch norm  
    ----
    -> relu
    -> max pool
    -> dropout
    ----
    -> conv
    -> relu
    -> conv
    -> batch norm
    ----
    -> relu
    -> max pool
    -> dropout
    -> flatten
    ----
    -> fully connected
    -> batch norm
    ----
    -> relu
    ----
    -> fully connected
    -> batch norm
    ----
    -> relu
    ----
    -> fully connected
    -> batch norm
    ----
    -> relu
    -> dropout
    ----
    -> fully connected
    -> softmax

"""
class CNN(torch.nn.Module):
    # TODO: define model's layers
    def __init__(self):
        super(CNN, self).__init__()
        pass

    # TODO: define model's forward pass
    def forward(self, x: torch.Tensor):
        pass

# TODO: implement training function
def train(model: nn.Module, X_data: torch.Tensor, Y_data: torch.Tensor, batch_size: int = 128, iterations: int = 1000):
    loss_function = nn.NLLLoss(reduction='none') 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0) 
    losses, accuracies = [], []
    pass

# TODO: implement testing function
def test(model: nn.Module, X_data: torch.Tensor, Y_data: torch.Tensor):
    pass

# TODO: implement plotting function to visualize model's performance
def plot(model: nn.Module, X_data: torch.Tensor, Y_data: torch.Tensor):
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

    # standardize feature set
    mean_px = X_train.mean().astype(np.float32)
    std_px = X_train.std().astype(np.float32)
    X_train = (X_train - mean_px) / (std_px)

    # load training and test sets into torch tensors
    X_train = torch.from_numpy(X_train.copy()).float()
    Y_train = torch.from_numpy(Y_train.copy())
    X_test  = torch.from_numpy(X_test.copy()).float()
    Y_test  = torch.from_numpy(Y_test.copy())

    n_classes = torch.unique(Y_train).numel()
    print(X_train[0])

    # DIAGNOSTIC PRINTS 
    print(f'n_classes: {n_classes}')
    print(f'X_train: {X_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'Y_train: {Y_train.shape}')
    print(f'Y_test: {Y_test.shape}')

    print(f'training model on {device}...')

    print('COMPLETE: convolutional neural network')

# run the entire program
if __name__ == '__main__':
    main()
