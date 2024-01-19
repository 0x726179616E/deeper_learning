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