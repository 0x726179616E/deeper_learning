#!/usr/bin/env python3 

from engine import *

# round to 1 decimal places when printing numpy array values 
np.set_printoptions(precision=3)

# test out the "autograd engine"
def driver():
    # create nodes
    x = Node(np.array([2.0, 3.0]))
    y = Node(np.array([4.0, 2.5]))
    b = Node(np.array([-0.5, 0.5]))

    # forward pass
    w = add(mul(x,y), b) # (2*4 - 0.5, 3*2.5 + 0.5)
    z = softmax(w) # (0.377, 0.622)

    # backward pass
    z.backward()

    # print values within each node
    print(f'x = \n{x.value}\n')
    print(f'y = \n{y.value}\n')
    print(f'b = \n{b.value}\n')
    print(f'w = (x * y) + b =\n{w.value}\n')
    print(f'z = softmax(w) =\n{z.value}\n')

    # print gradients within each node
    print()
    print(f'gradient wrt x = \n{x.grad}\n')
    print(f'gradient wrt y = \n{y.grad}\n')
    print(f'gradient wrt b = \n{b.grad}\n')
    print(f'gradient wrt w = \n{w.grad}\n')
    print(f'gradient wrt z = \n{z.grad}\n')
    return -1

# run program
if __name__ == "__main__":
    if driver() != -1:
        raise RuntimeError