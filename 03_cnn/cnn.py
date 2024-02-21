#!/usr/bin/env python3

import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

# detect torch device
if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# define convnet 
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        # fully connected layer
        self.fc = nn.Linear(in_features=576, out_features=10)
        
    def forward(self, x: torch.Tensor):
        # input [BS x 1 @ 28 x 28] where BS = 64
        x = F.relu(self.conv1(x)) # [BS x 32 @ 24 x 24]
        x = F.relu(self.conv2(x)) # [BS x 32 @ 20 x 20]
        x = self.bn1(x)
        x = self.pool(x) # [BS x 32 @ 10 x 10]
        x = F.relu(self.conv3(x)) # [BS x 64 @ 8 x 8]
        x = F.relu(self.conv4(x)) # [BS x 64 @ 6 x 6]
        x = self.bn2(x)
        x = self.pool(x) # [BS x 64 @ 3 x 3]
        x = x.view(-1, 576) # [BS x 576]
        x = self.fc(x) # [BS x 10]
        return x

# train the model
def train(model: torch, X_data: torch.Tensor, Y_data: torch.Tensor, epochs=30):
    loss_function = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
    BS = 64 # batch size

    # training loop
    for epoch in range(epochs):
        print(f"epoch: {epoch + 1}")
        for _ in (t := tqdm.trange(937)): # 60000 // BS
            # randomly sample the minibatch
            sample = np.random.randint(low=0, high=X_data.size(0), size=(BS))
            X = X_data[sample]
            Y = Y_data[sample]

            optimizer.zero_grad() # flush out precious loop's gradients
            Y_pred = model(X) # forward pass
            loss = loss_function(Y_pred, Y).mean() # compute loss
            loss.backward() # backward pass
            optimizer.step() # update weights
            t.set_description(f"loss: {loss:.5f}") # display current epoch's loss

# evaluate the model
def test(model: torch, X_data: torch.Tensor, Y_data: torch.Tensor):
    Y_preds = model(X_data).long()
    return (Y_preds.argmax(axis=1) == Y_data).float().mean()


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
    X_train = load('../data/mnist/train-images-idx3-ubyte')[0x10:].reshape((-1,1,28,28))
    Y_train = load('../data/mnist/train-labels-idx1-ubyte')[8:]
    X_test = load('../data/mnist/t10k-images-idx3-ubyte')[0x10:].reshape((-1,1,28,28))
    Y_test = load('../data/mnist/t10k-labels-idx1-ubyte')[8:]

    # convert training and test sets into torch tensors 
    X_train_tensor = torch.from_numpy(X_train.copy()).to(device).float()
    Y_train_tensor = F.one_hot(torch.from_numpy(Y_train.copy()).to(device).long()).float()
    X_test_tensor = torch.from_numpy(X_test.copy()).to(device).float()
    Y_test_tensor = torch.from_numpy(Y_test.copy()).to(device)

    # instantiate the model
    model = CNN().to(device)
    print(f'\ntraining model on {device}...\n')
    train(model, X_train_tensor, Y_train_tensor)
    result = test(model, X_test_tensor, Y_test_tensor)
    print(f"\ntest accuracy: {result.item() * 100:.2f}%\n") # test accuracy should be ~98-99%

    print('COMPLETE: convolutional neural network')

# run the entire program
if __name__ == '__main__':
    main()
