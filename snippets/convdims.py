#!/usr/bin/env python3

""" calculate conv layer output's spatial dimension (width or height) """
def conv_out(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2*padding) / stride + 1

""" calculate pooling layer output's spatial dimension (width or height) """
def pool_out(input_size, kernel_size=2, stride=2, padding=0):
    return (input_size - kernel_size + 2*padding) / stride + 1

W = 28
F = 5
W1 = conv_out(W, F)
print(f"conv1: {W1}")

W2 = conv_out(W1, F)
print(f"conv2: {W2}")

W3 = pool_out(W2)
print(f"pool1: {W3}")

F = 3
W4 = conv_out(W3, F)
print(f"conv3: {W4}")

W5 = conv_out(W4, F)
print(f"conv4: {W5}")

W6 = pool_out(W5)
print(f"pool2: {W6}")
