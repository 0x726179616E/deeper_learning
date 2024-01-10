#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import math

"""
multinomial logistic regression (aka softmax regression): https://en.wikipedia.org/wiki/Multinomial_logistic_regression
"""
class LogisticRegression:
    # initialize model
    def __init__(self, lr=0.01, iters=1000):
        self.w = None            # weights
        self.b = None            # bias
        self.learning_rate = lr  # learning rate
        self.iterations = iters  # iterations

    # fit model to training set
    def fit(self, X, y):
        n, d = X.shape # number of samples and features, respectively
        k = y.shape[1] # number of classes
        self.weights = np.zeros((d, k))
        self.bias = np.zeros((n, k))

        # gradient descent loop
        for _ in range(self.iterations):
            scores = np.dot(X, self.weights) + self.bias # compute linear transformation
            preds = self.softmax(scores) # compute nonlinearity 

            # compute gradients
            dw = (1 / n) * np.dot(X.T, (preds - y))
            db = (1 / n) * np.sum(preds - y, axis=1)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # predict on test set
    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias # compute linear transformation 
        preds = self.softmax(scores) # compute nonlinearity (converts logits to probabilities)
        return np.argmax(preds, axis=1) 

    def softmax(self, z):
        exp = np.exp(z - np.max(z))
        return exp / exp.sum(axis=1, keepdims=True)

# np arr -> shuffle -> split into training and test -> split into features and targets -> one-hot encode targets
def split_data(arr, test_sz=0.20): 
    if not 0 < test_sz < 1: raise ValueError("test size must be a float value between 0 and 1")

    np.random.shuffle(arr) # shuffle np array
    n = arr.shape[0] # number of samples
    i = int(n * (1-test_sz)) # split index

    # remove first column from array (as it just contains the index for each sample)
    arr = arr[:, 1:]

    # split into training and testing sets
    train_data, test_data = arr[:i], arr[i:]

    # pull targets out of the last column
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # convert target strings to numerical labels
    unique_classes = np.unique(arr[:, -1])
    label_to_id = {label: idx for idx, label in enumerate(unique_classes)}
    y_train_ids = np.array([label_to_id[label] for label in y_train])
    y_test_ids = np.array([label_to_id[label] for label in y_test])

    # one hot encode the target vectors 
    y_train = np.eye(len(unique_classes))[y_train_ids]
    y_test = np.eye(len(unique_classes))[y_test_ids]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # get csv file's absolute path
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/iris.csv')
    # load data into numpy array
    arr = pd.read_csv(f).to_numpy()
    X_train, X_test, y_train, y_test = split_data(arr)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("COMPLETE: multinomial logistic regression")


