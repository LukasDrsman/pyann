import numpy as np
import math

# sigmoid definitions
def sigmoid_uni(x):
    return 1 / 1 + math.exp(-x)

def sigmoid_diff(x):
    return sigmoid_uni(x) * (1 - sigmoid_uni(x))

sigmoid = np.vectorize(sigmoid_uni)

# neural network class
class ANN:
    def __init__(self, n, m, in, out):
        # save network's parameters
        self.n     = n
        self.m     = m
        self.in   = in
        self.out = out

        # generate random neural network
        self.W  = np.random.rand(m - 1, n, n)
        self.b  = np.random.rand(m - 1, n)
        self.IW = np.random.rand(n, in)
        self.OW = np.random.rand(out, n)
        self.Ib = np.random.rand(n)
        self.Ob = np.random.rand(out)

    def process(self, i):
        # compute first hidden layer
        N = np.zeros((self.m, self.n))
        N[0,:] = sigmoid(self.IW.dot(i) - self.Ib)

        # compute remaining hidden layers
        for i in range(self.m - 1):
            N[i + 1,:] = sigmoid(self.W[i].dot(N[i,:]) - self.b[i])

        # compute the output layer
        o = sigmoid(self.OW.dot(N[self.m - 1,:]) - self.Ob)

        # return o & N (for backpropagation)
        return o, N

    def backprop(self, X, iter, rt):
        # TODO: implement backpropagation function
        pass
