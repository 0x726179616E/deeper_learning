#!/usr/bin/env python3

import os
import numpy as np

class NN:
    def __init__(self, learning_rate=0.001, batch_size=128):
        self.w1 = self._init_layer(784, 128) # input layer to hidden layer
        self.w2 = self._init_layer(128, 10) # hidden layer to output layer
        self.lr = learning_rate
        self.batch_size = batch_size

    # numerically stable normalization via logsoftmax
    def _logsoftmax(self, z):
        z -= np.max(z)
        lsm = z - np.log(np.sum(np.exp(z)))
        return lsm

    # derivative of logsoftmax 
    def _dlogsoftmax(self, z):
        sm = np.exp(self._logsoftmax(z))
        return np.diagflat(sm) - np.outer(sm, sm)

    # ReLU activation function
    def _relu(self, z):
        return np.maximum(z, 0)

    # derivative of ReLU
    def _drelu(self, z):
        return (z > 0).astype(float)

    # negative log likelihood loss
    def _nll(self, preds, targets):
        N = preds.shape[0]
        correct_preds = preds[np.arange(N).astype(int).reshape((-1,1)), targets.astype(int)]
        loss = -np.mean(correct_preds)
        return loss

    # derivative of NLL (targets must be one-hot encoded)
    def _dnll(self, logits, targets):
        probs = np.exp(self._logsoftmax(logits))
        N = logits.shape[0]
        grad = (probs - targets) / N
        return grad

    # layer's dimensions are specified by n (rows) and d (columns)
    def _init_layer(self, n, d):
        # initialize layer's weights via uniform distribution
        ret = np.random.uniform(-1.0, 1.0, size=(n,d))/np.sqrt(n*d)
        return ret.astype(np.float32)

    # training loop
    def _loop(self, X_data, Y_data):
        N = len(Y_data) # number of samples

        # one-hot encode the targets
        Y_onehot = np.zeros((N,10), np.float32)
        Y_onehot[range(Y_onehot.shape[0]), Y_data] = 1 

        print(f'X:                  {X_data.shape}')
        print(f'w1:                 {self.w1.shape}')
        print(f'Y:                  {Y_onehot.shape}')

        # forward pass
        z1 = X_data.dot(self.w1) # first linear transformation
        print(f'z1 = X @ w1:        {z1.shape}')

        l1 = self._relu(z1) # relu nonlinearity
        print(f'l1 = relu(z1):      {l1.shape}')

        print(f'w2:                 {self.w2.shape}')
        l2 = l1.dot(self.w2) # second linear transformation
        print(f'l2 = l1 @ w2:       {l2.shape}')

        z2 = self._logsoftmax(l2) # logsoftmax normalization
        print(f'z2 = logsoftmax(l2):{z2.shape}')

        loss = self._nll(z2, Y_onehot) # negative log likelihood loss function
        print(f'loss = nll(z2,Y)    {loss}')

        # TODO: backward pass
        # 1. dloss/dz2 
        # 2. dloss/dl2 = dloss/dz2 * dz2/dl2  
        # 3. dloss/dl1 = dloss/dl2 * dl2/dl1
        #    dloss/dw2 = dloss/dl2 * dl2/dw2
        # 4. dloss/dz1 = dloss/dl1 * dl1/dz1
        # 5. dloss/dw1 = dloss/dz1 * dz1/dw1

        dz2 = self._dnll(logits=z2, targets=Y_onehot) # grad through loss fn
        print(f'dz2: {dz2.shape}')

        dl2 = self._dlogsoftmax(z2) * dz2 # grad through logsoftmax
        print(f'dz2: {dz2.shape}')

        dl1 = dl2.dot(self.w2.T) # grad through second linear transformation
        print(f'dl1: {dl1.shape}')

        dw2 = dl2.dot(l1.T) # grad through second linear transformation
        print(f'dw2: {dw2.shape}')

        dz1 = dl1 * self._drelu(l1) # grad through relu
        print(f'dz1: {dz1.shape}')

        dw1 = dz1.dot(X_data)
        print(f'dw1: {dw1.shape}')
        pass
        # return loss, l2, dw1, dw2

    def train(self, X_data, Y_data, iterations=1):
        print('Training now...')
        for i in range(iterations):
            # print(f"loop {i}:")
            self._loop(X_data[ : self.batch_size], Y_data[ : self.batch_size])
            # loss, l2, dw1, dw2 = self._loop(X_data[ : self.batch_size], Y_data[ : self.batch_size])
        print()
        print("Done training.")


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
    X_train = fetch_data('../data/train-images-idx3-ubyte')[0x10:].reshape((-1, 28*28))
    Y_train = fetch_data('../data/train-labels-idx1-ubyte')[8:]
    X_test = fetch_data('../data/t10k-images-idx3-ubyte')[0x10:].reshape((-1, 28*28))
    Y_test = fetch_data('../data/t10k-labels-idx1-ubyte')[8:]
    
    model = NN()
    model.train(X_train, Y_train)

    print('COMPLETE: multilayer perceptron')

# run the program
if __name__ == '__main__':
    main()
