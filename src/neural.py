 #      ___           ___           ___           ___           ___           ___
 #     /\__\         /\  \         /\__\         /\  \         /\  \         /\__\
 #    /::|  |       /::\  \       /:/  /        /::\  \       /::\  \       /:/  /
 #   /:|:|  |      /:/\:\  \     /:/  /        /:/\:\  \     /:/\:\  \     /:/  /
 #  /:/|:|  |__   /::\~\:\  \   /:/  /  ___   /::\~\:\  \   /::\~\:\  \   /:/  /
 # /:/ |:| /\__\ /:/\:\ \:\__\ /:/__/  /\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/__/
 # \/__|:|/:/  / \:\~\:\ \/__/ \:\  \ /:/  / \/_|::\/:/  / \/__\:\/:/  / \:\  \
 #     |:/:/  /   \:\ \:\__\    \:\  /:/  /     |:|::/  /       \::/  /   \:\  \
 #     |::/  /     \:\ \/__/     \:\/:/  /      |:|\/__/        /:/  /     \:\  \
 #     /:/  /       \:\__\        \::/  /       |:|  |         /:/  /       \:\__\
 #     \/__/         \/__/         \/__/         \|__|         \/__/         \/__/.py

import numpy as np
from math import exp

def sig(x):
    try:
        a = 1 / (1 + exp(-x))
        return a
    except:
        if x < 0:
            return 0
        else:
            return 1

sigv = np.vectorize(sig)

class ANN:
    def __init__(self, n, l, i, o):
        self.W = []
        self.b = []
        self.W.append(np.random.rand(n, i) - 0.5)
        self.b.append(np.random.rand(n) - 0.5)
        for _ in range(l - 1):
            self.W.append(np.random.rand(n, n) - 0.5)
            self.b.append(np.random.rand(n) - 0.5)
        self.W.append(np.random.rand(o, n) - 0.5)
        self.b.append(np.random.rand(o) - 0.5)

    def compute(self, x):
        o = []
        o.append(x)
        for i in range(0, len(self.W)):
            o.append(sigv(self.W[i].dot(o[i]) - self.b[i]))

        return o

    def backpropagate(self, X, iterations, eta, outlev=True):
        for c in range(iterations):
            deltas = []
            avg = 1 / len(X)
            O = [self.compute(x[0]) for x in X]
            for o, x in zip(O, X):
                oc = o[-1]
                t  = x[1]
                deltas.append(
                    avg * 2 * (oc - t) * oc * (-oc + 1)
                )

            dW = len(self.W) * [0]
            db = len(self.b) * [0]
            for l in range(len(self.W) - 1, -1, -1):
                dW[l] = np.outer(deltas[0], O[0][l])
                db[l] = eta * sum(deltas)

                for i in range(1, len(deltas)):
                    dW[l] += np.outer(deltas[i], O[i][l])

                dW[l] *= -eta

                for i in range(len(deltas)):
                    deltas[i] = self.W[l].transpose().dot(deltas[i])

            for i in range(len(self.W)):
                self.W[i] = self.W[i] + dW[i]
                self.b[i] = self.b[i] + db[i]

            if outlev:
                print(self.W)
                print(self.b)

            for x in X:
                o = self.compute(x[0])[-1]
                if outlev: print(o)
