#!/usr/bin/env python3

""" calculate conv layer output's spatial dimension (width or height) """
def conv_out(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2*padding) / stride + 1

""" calculate pooling layer output's spatial dimension (width or height) """
def pool_out(input_size, kernel_size=2, stride=2, padding=0):
    return (input_size - kernel_size + 2*padding) / stride + 1
