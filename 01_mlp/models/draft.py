#!/usr/bin/env python3

import numpy as np
from tqdm import trange
import os

class MultilayerPerceptron:
    def __init__(self, layer_sizes, learning_rate=0.1, activation_function='sigmoid'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.activation_function = activation_function

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _forward(self, x):
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self._sigmoid(z)
            activations.append(activation)

        return activations, zs

    def _backward(self, x, y, activations, zs):
        delta = self._cost_derivative(activations[-1], y) * \
                self._sigmoid_derivative(zs[-1])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # For the last layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # For the rest of the layers
        for l in range(2, len(self.layer_sizes)):
            z = zs[-l]
            sp = self._sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def _update_mini_batch(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            activations, zs = self._forward(x)
            delta_nabla_b, delta_nabla_w = self._backward(x, y, activations, zs)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (self.learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def _cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def train(self, training_data, epochs, mini_batch_size):
        n = len(training_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch)

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
