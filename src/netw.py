import numpy as np
from props import *
from math import exp

############################
#
# math
#
############################

def sig(x):
    a = 0
    try:
        a = 1 / (1 + exp(-x))
    except:
        if x < 0:
            a = 0
        else:
            a = 1
    return a


sigv = np.vectorize(sig)

def compute(x, W):
    o = []
    o.append(x)

    for i in range(0, len(W)):
        o.append(sigv(W[i].dot(o[i])))

    return o

############################
#
# new ANN
#
############################

def new(conf):
    W = []
    W.append((np.random.rand(conf.neurons, conf.io["i"]) - 0.5) * 10)
    for i in range(conf.layers - 1):
        print(i)
        W.append((np.random.rand(conf.neurons, conf.neurons) - 0.5) * 10)
    W.append((np.random.rand(conf.io["o"], conf.neurons) - 0.5) * 10)

    return W

############################
#
# backpropagate
#   -> recurse over deltas
#   -> construct W diff matrix
#
############################

def backpropagate(W, X, iterations, eta):
    for c in range(iterations):
        deltas = []
        avg = 1 / len(X)
        O = [compute(x[0], W) for x in X]
        for o, x in zip(O, X):
            oc = o[-1]
            t  = x[1]
            deltas.append(
                avg * 2 * (oc - t) * oc * (np.ones(oc.shape) - oc)
            )

        dW = len(W) * [0]
        for l in range(len(W) - 1, -1, -1):
            dW[l] = np.outer(deltas[0], O[0][l])
            for i in range(1, len(deltas)):
                dW[l] = dW[l] + np.outer(deltas[i], O[i][l])

            dW[l] = -eta * dW[l]

            for i in range(len(deltas)):
                deltas[i] = W[l].transpose().dot(deltas[i])

        for i in range(len(W)):
            W[i] = W[i] + dW[i]

        print(W)
        for x in X:
            o = compute(x[0], W)[-1]
            print(o)
