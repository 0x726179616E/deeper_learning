#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os 

""" 
multinomial logistic regression (aka softmax regression): https://en.wikipedia.org/wiki/Multinomial_logistic_regression
"""
class LogisticRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.w = None      # weights
        self.b = None      # biases
        self.lr = lr       # learning rate
        self.iters = iters # iterations
        
    # fit model to training data
    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape # number of samples and features, respectively
        k = y.shape[1] # number of output classes
        self.w = np.zeros((d, y.shape[1])) # init weights
        self.b = np.zeros(y.shape[1]) # init biases

        # training loop
        for _ in range(self.iters):
            # forward pass
            z = np.dot(X, self.w) + self.b # compute score
            y_pred = softmax(z) # compute probablities

            # compute gradients
            dw = np.dot(X.T, (y_pred - y)) / n
            db = np.sum(y_pred - y, axis=0) / n
            
            # update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    # make predictions on test data
    def predict(self, X: np.ndarray):
        z = np.dot(X, self.w) + self.b  
        return np.argmax(softmax(z), axis=1) 

# softmax activation function
def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

# file -> pandas df -> one hot encode -> numpy arr -> shuffle -> split features and labels -> split training and test sets
def prepare_data(file, test_sz=0.21):
    if not 0 < test_sz < 1: raise ValueError("test size must be a float between 0 and 1")

    # load in csv file and one hot encode the target classes
    data = pd.get_dummies(pd.read_csv(file), columns=['Species']).to_numpy()[:,1:]
    np.random.shuffle(data) # shuffle entire dataset
    X = data[:, :-3] # pull out features 
    y = data[:, -3:] # pull out targets

    # split features and targets into training and test sets
    split_index = int(1.0 - (test_sz * data.shape[0]))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

def plot(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray):
    # get predictions
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)

    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"accuracy on test set: {accuracy * 100:.2f}%")

    # plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    unique_species = ['setosa', 'versicolor', 'virginica']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_species, yticklabels=unique_species)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


if __name__ == "__main__":
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/iris.csv')
    X_train, X_test, y_train, y_test = prepare_data(f)
    
    # train model
    model = LogisticRegression(lr=0.01, iters=1500)
    model.fit(X_train, y_train)

    # test model
    pred = model.predict(X_test)
    
    # plot model performance
    plot(model, X_test, y_test)

    print("COMPLETE: multinomial logistic regression")