#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

""" 
multiple linear regression model: https://en.wikipedia.org/wiki/Linear_regression#Simple_and_multiple_linear_regression 
 """
class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        # add column of ones for the bias term (intercept)
        X = np.column_stack((np.ones(X.shape[0]), X)) 
        # using the normal equation for a closed-form solution: W = (X'X)^(-1)(y'X)
        params = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(y.T, X))
        self.b = params[0]
        self.w = params[1:]

    def predict(self, X):
        if self.w is None: raise Exception("model is not trained yet")
        X = np.column_stack((np.ones(X.shape[0]), X)) 
        return np.dot(X, np.insert(self.w, 0, self.b))

# np arr -> shuffle -> split into training and test -> split into features and targets
def split_data(arr, test_sz=0.20):
    if not 0 < test_sz < 1: raise ValueError("test size must be a float between 0 and 1")

    np.random.shuffle(arr) # shuffle array
    n = arr.shape[0] # number of samples
    i = int(n * (1 - test_sz)) # split index

    # split into training and testing sets
    train_data, test_data = arr[:i], arr[i:]

    # pull targets out of the last column 
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
     
    return X_train, X_test, y_train, y_test

# plot predictions vs targets
def plot(y_target, y_pred, title="Targets vs Predicted"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_target, y_pred, alpha=0.7, color='blue')
    # plt.scatter(y_target, y_pred, color='blue')
    plt.plot([y_target.min(), y_target.max()], [y_target.min(), y_target.max()], color='red')  # Diagonal line
    plt.xlabel('Targets')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # get csv file's absolute path
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/housing.csv')
    # prepare data
    arr = pd.read_csv(f).loc[:, ["Beds", "Baths", "Living Space", "Price"]].to_numpy()
    X_train, X_test, y_train, y_test = split_data(arr)

    model = LinearRegression()         # instantiate model
    model.fit(X_train, y_train)        # train model
    pred = model.predict(X_test)       # test model
    plot(y_test, pred)                 # display results 
    print("COMPLETE: multiple linear regression")
