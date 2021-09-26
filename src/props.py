import numpy as np

eXample = [ (np.array([2, 3, 4, 2, 3, 4, 2, 3, 4]), np.array([0.9, 0.1])),
            (np.array([1, 5, 2, 1, 5, 2, 1, 5, 2]), np.array([0.1, 0.9])),
            (np.array([6, 0, 2, 6, 0, 2, 6, 0, 2]), np.array([0.2, 0.2])) ]


class DefANN:
    neurons = 4
    layers = 2
    eta = 5.0
    div = 0.0
    iterations = 30000000
    io = {
        "i" : 9,
        "o" : 2
    }
