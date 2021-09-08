import numpy as np

def main():
    # test neural.py
    test = ANN(16, 3, 3, 2)
    image = np.random.rand(3, 3)

    # generate test data
    #   -> (pos : np (x, y), input : np (c1, c2, c3))
    data = np.array([])

    # train the neural network
    test.train(data, 100, 2)



if __name__ == "__main__":
    main()
