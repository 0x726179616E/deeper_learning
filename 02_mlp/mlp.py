#!/usr/bin/env python3

import os
import tqdm 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

# detect torch device
if torch.cuda.is_available(): device = 'cuda' 
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# multilayer perceptron class
class MLP(torch.nn.Module):
    def __init__(self, hidden_size=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 10, bias=False)
        self.sm = nn.LogSoftmax(dim=1)

    # forward pass
    def forward(self, x):
        x = self.fc1(x) # first linear layer
        x = F.relu(x)  # relu nonlinearity 
        x = self.fc2(x) # second linear layer
        x = self.sm(x) # logsoftmax normalization of outputs
        return x

# training function
def train(model: nn.Module, X_data: torch.Tensor, Y_data: torch.Tensor, batch_size=128, iterations=1000):
    loss_function = nn.NLLLoss(reduction='none') 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0) 
    losses, accuracies = [], []

    # training loop
    for i in (t := tqdm.trange(iterations)):
        # randomly sample the minibatch
        sample = np.random.randint(0, X_data.shape[0], size=(batch_size))
        X = X_data[sample]
        Y = Y_data[sample]

        model.zero_grad() # flush out previous loop's gradients
        out = model(X) # forward pass
        cat = torch.argmax(out, dim=1) # get categories
        accuracy = (cat == Y).float().mean() # compute accuracy
        loss = loss_function(out, Y).mean() # compute loss
        loss.backward() # backward pass
        optimizer.step() # update weights

        losses.append(loss.item())
        accuracies.append(accuracy.item())
        t.set_description(f'loss: {loss:.2f} | training accuracy: {accuracy * 100:.2f}%')
    plt.ylim(-0.1, 1.2)
    plt.plot(losses)
    plt.plot(accuracies)

# testing function
def test(model: nn.Module, X_data: torch.TensorType, Y_data: torch.Tensor):
    Y_preds = torch.argmax(model(X_data), dim=1).float()
    return (Y_data == Y_preds).float().mean()

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
    X_train = load('../data/mnist/train-images-idx3-ubyte')[0x10:].reshape((-1, 28*28))
    Y_train = load('../data/mnist/train-labels-idx1-ubyte')[8:]
    X_test = load('../data/mnist/t10k-images-idx3-ubyte')[0x10:].reshape((-1, 28*28))
    Y_test = load('../data/mnist/t10k-labels-idx1-ubyte')[8:]

    # load training and test sets into torch tensors
    X_train = torch.from_numpy(X_train.copy()).to(device).float()
    Y_train = torch.from_numpy(Y_train.copy()).to(device).long()
    X_test  = torch.from_numpy(X_test.copy()).to(device).float()
    Y_test  = torch.from_numpy(Y_test.copy()).to(device).long()

    # instantiate model 
    model = MLP().to(device)
    print(f'training model on {device}...')
    train(model, X_train, Y_train, batch_size=128, iterations=3000)
    result = test(model, X_test, Y_test)
    print(f'test accuracy: {result.item() * 100:.2f}%') # should achieve ~95% test accuracy

    print('COMPLETE: multilayer perceptron')

# run the entire program
if __name__ == '__main__':
    main()
