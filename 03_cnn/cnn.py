#!/usr/bin/env python3

import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, bias=False)

        # fully connected layers
        self.fc1 = nn.Linear(1024, 256, bias=False) # 1024 = 64(BS)x4x4
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.fc3 = nn.Linear(128, 84, bias=False)
        self.fc4 = nn.Linear(84, 10, bias=False)

        # pooling layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # regularization layers 
        self.dropout = nn.Dropout(0.25)

        # normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(84)

    def forward(self, x: torch.Tensor):
        # feature maps' dimensions: channels @ width x height
        x = F.relu(self.conv1(x)) # 32 @ 32 x 32
        x = F.relu(self.conv2(x)) # 32 @ 28 x 28
        x = F.relu(self.bn1(x))
        x = self.maxpool(self.dropout(x)) # 32 @ 14 x 14
        x = F.relu(self.conv3(x)) # 64 @ 12 x 12
        x = F.relu(self.conv4(x)) # 64 @ 10 x 10
        x = F.relu(self.bn2(x))
        x = self.maxpool(self.dropout(x)) # 64 @ 4 x 4
        x = x.view(x.size(0), -1) # 64 x 1024
        x = F.relu(self.fc1(x)) # 64 x 256
        x = self.bn3(x)
        x = F.relu(self.fc2(x)) # 64 x 128
        x = self.bn4(x)
        x = F.relu(self.fc3(x)) # 64 x 84
        x = self.bn5(x)
        x = self.fc4(x) # 64 x 10
        # x = F.softmax(x, dim=1)    
        return x

# load mnist dataset from ubyte file into np array
def load(file_path):
    abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    try: 
        with open(abs_file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    except IOError: 
        print(f'error reading file: {file_path}')
    return data

# train the model
def train(model: nn.Module, training_set: DataLoader, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
    for epoch in (t := tqdm.trange(epochs)):
        for X, Y in training_set:
            optimizer.zero_grad() # flush out previous loop's gradients
            out = model(X) # forward pass
            loss = criterion(out, Y.float()) # compute loss
            loss.backward() # backward pass
            optimizer.step() # update weights
            # display progress bar 
            t.set_description(f"epoch {epoch} loss: {loss.item():.6f}")
            # t.refresh()

# evaluate the model
def test(model: nn.Module, test_set: DataLoader):
        # model.eval()
        # with torch.no_grad():
        #     # compute cost function
        #     validation_loss = sum(criterion(model(X_val), Y_val) for X_val, Y_val in test_loader)
        #     scheduler.step(validation_loss)

        # # evaluate model's performance on test set 
        # model.eval()
        # test_loss = 0
        # correct = 0
        # with torch.no_grad():
        #     for X_test, Y_test in test_loader:
        #         out = model(X_test)
        #         test_loss += criterion(out, Y_test).item()
        #         pred = out.data.max(1, keepdim=True)[1]
        #         correct += pred.eq(Y_test.data.view_as(pred)).sum()
        # test_loss /= len(test_loader.dataset)
        # print(f"test set: average loss: {test_loss:.4f}, accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct/len(test_loader.dataset):0.f}%)")
    pass

# driver function
def main():
    # load training set and test set into numpy ararys
    X_train = load('../data/mnist/train-images-idx3-ubyte')[0x10:].reshape((-1,28,28))
    Y_train = load('../data/mnist/train-labels-idx1-ubyte')[8:]
    X_test = load('../data/mnist/t10k-images-idx3-ubyte')[0x10:].reshape((-1,28,28))
    Y_test = load('../data/mnist/t10k-labels-idx1-ubyte')[8:]

    # standardize examples
    mean_px =  X_train.mean().astype(np.float32)
    stddev_px =  X_train.std().astype(np.float32)
    X_train = (X_train - mean_px) / stddev_px
    X_test = (X_test - mean_px) / stddev_px

    # normalize images and pad them from 28x28 to 32x32
    X_train_tensor = F.pad(torch.tensor(X_train / 255.).unsqueeze(1).float(), (2,2,2,2), "constant", 0).to(device)
    X_test_tensor = F.pad(torch.tensor(X_test / 255.).unsqueeze(1).float(), (2,2,2,2), "constant", 0).to(device)
    # one hot encode the labels
    Y_train_tensor = F.one_hot(torch.tensor(Y_train).long(), num_classes=10).to(device)
    Y_test_tensor = F.one_hot(torch.tensor(Y_test).long(), num_classes=10).to(device)

    # create tensor datasets
    training_set_tensor = TensorDataset(X_train_tensor, Y_train_tensor)
    test_set_tensor = TensorDataset(X_test_tensor, Y_test_tensor)

    # create data loaders
    training_set = DataLoader(training_set_tensor, batch_size=64, shuffle=True)
    test_set = DataLoader(test_set_tensor, batch_size=64, shuffle=False)

    # instantiate model, optimizer, c
    model = CNN().to(device)
    print(f'\ntraining model on {device}...\n')
    train(model, training_set=training_set, epochs=5)

    print('COMPLETE: convolutional neural network')

# run the entire program
if __name__ == '__main__':
    main()
