#!/usr/bin/env python3

import os
import numpy as np
import torch

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
