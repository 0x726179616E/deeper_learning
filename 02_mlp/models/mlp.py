#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

# Neural Network for MNIST 
# - model hyperparameters:
#       - learning rate: 0.1 -> 0.001 (consider step/exponential decay for lr)
#       - batch size: 64 or 128 samples a time 
#       - training epochs: 10-20
class NN:
    def __init__(self, layers, learning_rate=0.1, batch_size=64, epochs=15, weight_decay=0.001):
        # load model dimensions 
        self.layers = layers 

        # initialize hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay

        # randomly initialize model's weights and biases
        self.weights = [np.random.randn(y,x) for x,y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y,1) for y in layers[1:]]
    
    # ReLU method
    def _relu(self, z):
        return np.maximum(0,z)

    # ReLU derivative method
    def _drelu(self, z):
        return (z > 0).astype(float)
    
    # softmax method
    def _softmax(self, z):
        exps = np.exp(z - np.max(z)) # improve numerical stability
        return exps / np.sum(exps, axis=0, keepdims=True)

    # forward pass
    def _forward(self, x):
        zs = [] # list to store all the score vectors, layer by layer
        activation = x
        activations = [x] # list to store all the activations

        # forward pass through hidden layers (use ReLU as activation functoin)
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            print(f'w.shape = {w.shape}')
            print(f'a.shape = {activation.shape}')
            print(f'b.shape = {b.shape}')
            print('attempting: w @ a + b')
            activation = activation.reshape(784, 1)
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self._relu(z)
            activations.append(activation)

        # forward pass through output layer (use softmax activation function)
        print(f'output w.shape = {self.weights[-1].shape}')
        print(f'output a.shape = {activation.shape}')
        print(f'output b.shape = {self.biases[-1].shape}')

        print("\nPRINTING WEIGHT DIMENSIONS")
        i = 0
        for l in self.weights:
            print(f'l{i}.shape = {l.shape}')
            i += 1
        print()


        z = np.dot(self.weights[:-1], activation) + self.biases[:-1]
        zs.append(z)
        activation = self._softmax(z)
        activations.append(activation)

        return activations, zs

    # backward pass
    def _backward(self, x, y, activations, zs):
        delta = activations[-1] - y

        # zero the gradients
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]

        # output layer
        db[-1] = delta
        dw[-1] = np.dot(delta, activations[-2].T)

        # remaining layers
        for l in range(2, len(self.layers)):
            z = zs[-1]
            sp = self._drelu(z)
            delta = np.dot(self.weights[- l + 1].T, delta) * sp
            db[-l] = delta
            dw[-l] = np.dot(delta, activations[- l + 1].T)

        return db, dw

    def _update_mini_batch(self, mini_batch):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            activations, zs = self._forward(x)
            delta_db, delta_dw = self._backward(x, y, activations, zs)
            db = [nb + dnb for nb, dnb in zip(db, delta_db)]
            dw = [nw + dnw for nw, dnw in zip(dw, delta_dw)]

            # apply weight decay
            self.weights = [(1 - self.weight_decay) * w - (self.learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, dw)]
            self.biases = [b - (self.learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, db)]

    # fit on training set
    ### TODO: EDIT THIS METHOD TO TAKE IN X_TRAIN AND Y_TRAIN AS TWO SEPERATE PARAMETERS AND NOT AS A SINGLE TRAINING SET PARAM
    def train(self, training_data):
        n = len(training_data) # number of samples

        for j in range(self.epochs):
            np.random.shuffle(training_data)
            mini_batches =  [training_data[k:k + self.batch_size] for k in range(0, n, self.batch_size)]

            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch)

    # predict on test set
    def predict(self, x):
        activations, _ = self._forward(x)
        return activations[-1]

# load mnist dataset from ubyte file into np array
def fetch_data(file_path):
    abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    try: 
        with open(abs_file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    except IOError: 
        print(f'error reading file: {file_path}')
    return data

# evaluate model's accuracy on test set 
def evaluate(model, test_data):
    results = [(np.argmax(model.predict(x)), y) for (x,y) in test_data]
    accuracy = sum(int(x==y) for (x,y) in results) / len(test_data)
    return accuracy

# driver function
def main():
    # load training set and test set
    X_train = fetch_data('../data/train-images-idx3-ubyte')[0x10:].reshape((-1, 28, 28))
    Y_train = fetch_data('../data/train-labels-idx1-ubyte')[8:]
    X_test = fetch_data('../data/t10k-images-idx3-ubyte')[0x10:].reshape((-1, 28, 28))
    Y_test = fetch_data('../data/t10k-labels-idx1-ubyte')[8:]

    ## print(f'X_train.shape: {X_train.shape}')
    ## print(f'Y_train.shape: {Y_train.shape}')
    ## print(f'X_test.shape: {X_test.shape}')
    ## print(f'Y_test.shape: {Y_test.shape}')

    ## print(X_test[5000])
    ## print(Y_test[5000])

    training_set = list(zip(X_train, Y_train))
    test_set = list(zip(X_test, Y_test))
    
    # model dimensions: 784 neuron input layer, 128 neuron hidden layer, 10 neuron output layer
    layer_sizes = [784, 128, 10]

    # # instantiate model
    model = NN(layers=layer_sizes, learning_rate=0.01, batch_size=64, epochs=15, weight_decay=0.001)
    model.train(training_data=training_set)
    accuracy = evaluate(model=model, test_data=test_set)
    print(f"model accuracy is: {accuracy}")
    
    print('COMPLETE: multilayer perceptron program')

# run the program
if __name__ == '__main__':
    main()
