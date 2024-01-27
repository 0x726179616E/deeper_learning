#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# inspired by: https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=5)
        # fully connected (linear) layers
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # first conv layer
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # second conv layer
        x = x.view(-1, 1024) # flatten tensor
        x = F.relu(self.fc1(x)) # first linear layer
        x = self.fc2(x) # second linear layer
        x = F.log_softmax(x, dim=1) # logsoftmax normalization of outputs
        return x

def train(model: nn.Module, X_data: torch.Tensor, Y_data: torch.Tensor, batch_size=128, iterations=1000):
    loss_function = nn.NLLLoss(reduction='none') 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0) 
    losses, accuracies = [], []
    # TODO: complete training function

# TODO: implement testing function

# TODO: implement plotting function to visualize model's performance

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
    # load computations onto GPU if possible
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): 
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else: 
            device = 'mps'
    else: device = 'cpu'
    print(f'device: {device}\n')

    # load training set and test set into numpy ararys
    X_train = load('../data/train-images-idx3-ubyte')[0x10:].reshape((-1, 28*28))
    Y_train = load('../data/train-labels-idx1-ubyte')[8:]
    X_test = load('../data/t10k-images-idx3-ubyte')[0x10:].reshape((-1, 28*28))
    Y_test = load('../data/t10k-labels-idx1-ubyte')[8:]

    # load training and test sets into torch tensors
    X_train = torch.from_numpy(X_train.copy()).to(device).float()
    Y_train = torch.from_numpy(Y_train.copy()).to(device).long()
    X_test  = torch.from_numpy(X_test.copy()).to(device).float()
    Y_test  = torch.from_numpy(Y_test.copy()).to(device).long()

    print('COMPLETE: convolutional neural network')

# run the entire program
if __name__ == '__main__':
    main()
