import numpy as np

def parse(fname, i, t):
    X = []
    with open(fname, "r") as file:
        for line in file:
            if line[0] == "#": continue
            I = np.zeros([i])
            T = np.zeros([t])

            vals = line.split(",")

            offset = 0
            for n in range(i):
                offset += 1
                I[n] = float(vals[n].strip())
            for n in range(t):
                T[n] = float(vals[n + offset].strip())

            X.append((I, T))
    return X
