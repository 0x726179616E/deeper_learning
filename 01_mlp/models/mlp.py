#!/usr/bin/env python3

import numpy as np
from tqdm import trange
import os

class MultilayerPerceptron:
    def __init__():
        pass

# load mnist dataset from ubyte file into np array
def fetch_data(file_path):
    abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    try: 
        with open(abs_file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    except IOError: 
        print(f'error reading file: {file_path}')
    return data

# driver function
def main():
    # load training set and test set
    X_train = fetch_data('../data/train-images-idx3-ubyte')[0x10:].reshape((-1, 28, 28))
    Y_train = fetch_data('../data/train-labels-idx1-ubyte')[8:]
    X_test = fetch_data('../data/t10k-images-idx3-ubyte')[0x10:].reshape((-1, 28, 28))
    Y_test = fetch_data('../data/t10k-labels-idx1-ubyte')[8:]

    print(f'X_train.shape: {X_train.shape}')
    print(f'Y_train.shape: {Y_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'Y_test.shape: {Y_test.shape}')


    print(X_test[5000])
    print(Y_test[5000])
    print('OK')

# run the program
if __name__ == '__main__':
    main()
