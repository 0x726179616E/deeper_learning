#!/usr/bin/env python3 

import numpy as np

# data structure for a Node in the autograd engine's computational graph 
class Node:
    def __init__(self, value, parents=None):
        self.value = value
        self.grad = 0.0
        self.parents = [] if parents is None else parents

    def backward(self, grad=1.0):
        self.grad += grad
        for parent, local_grad in self._backward():
            parent.backward(self.grad * local_grad)

    # method to be overriden by specific classes for each operation
    def _backward(self):
        return []

# ADDITION OP
class AddNode(Node):
    def __init__(self, x, y):
        super().__init__(x.value + y.value, parents=[x,y])

    def _backward(self):
        return [(self.parents[0], 1.0), (self.parents[1], 1.0)]

def add(x, y):
    return AddNode(x,y)

# MULTIPLICATION OP
class MulNode(Node):
    def __init__(self, x, y):
        super().__init__(x.value * y.value, parents=[x,y])

    def _backward(self):
        return [(self.parents[0], self.parents[1].value), (self.parents[1], self.parents[0].value)]

def mul(x, y):
    return MulNode(x,y)

# RELU OP
class ReLUNode(Node):
    def __init__(self, x):
        super().__init__(np.maximum(0, x.value), parents=[x])

    def _backward(self):
        grad = np.where(self.value > 0, [1, 0])
        return [(self.parents[0], grad)]

def relu(x):
    return ReLUNode(x)

# SOFTMAX OP
class SoftMaxNode(Node):
    def __init__(self, x):
        exps = np.exp(x.value - np.max(x.value))
        super().__init__(exps / np.sum(exps), parents=[x])

    def _backward(self):
        s = self.value.reshape(-1, 1)
        grad = np.diagflat(s) - np.dot(s, s.T)
        return [(self.parents[0], grad)]

def softmax(x):
    return SoftMaxNode(x)

# test out the "autograd" engine 
def driver():
    # round to 2 decimal places when printing numpy array values 
    np.set_printoptions(precision=2)

    # create nodes
    x = Node(np.array([2.0, 2.0]))
    y = Node(np.array([3.0, 3.0]))
    b = Node(np.array([4.0, 4.0]))
    c = Node(np.array([8.0, 14.0]))

    # forward pass
    w = mul(add(x,y), c) 
    z = softmax(mul(w, b))

    # backward pass
    z.backward()

    # print values within each node
    print(f'x = \n{x.value}\n')
    print(f'y = \n{y.value}\n')
    print(f'b = \n{b.value}\n')
    print(f'c = \n{c.value}\n')
    print(f'w = (x + y) * c =\n{w.value}\n')
    print(f'z = softmax(w * b) =\n{z.value}\n')

    # print gradients within each node
    print()
    print(f'gradient wrt x = \n{x.grad}\n')
    print(f'gradient wrt y = \n{y.grad}\n')
    print(f'gradient wrt b = \n{b.grad}\n')
    print(f'gradient wrt c = \n{c.grad}\n')
    print(f'gradient wrt w = \n{w.grad}\n')
    print(f'gradient wrt z = \n{z.grad}\n')
    return 0

# run program
if __name__ == "__main__":
    if driver() != 0:
        raise RuntimeError