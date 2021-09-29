# pyann
Testing repository for TatranskiDravci/moses/hannah

## Structure
```
.
└── src
    ├── neural.py
    └── props.py
```
- `neural.py` - neural network structure and methods
- `props.py` - additional resources, such as training samples

## Usage
`neural.py` and `props.py` can be imported using
```py
import neural
import props
```
### Using `neural.py`
A neural network with 16 neurons per hidden, 2 hidden layers, 3 input neurons and 2 output neurons can be constructed using the `neural.ANN` constructor as such:
```py
import neural

ann = neural.ANN(16, 2, 3, 2)
```
The `neural.ANN.compute` method handles data processing. A one-dimensional numpy array is fed to this function, with its shape corresponding with the input layer of the network. The network outputs a list of activations for all layers, with the last activation element corresponding with the output layer:
```py
import numpy as np
import neural

ann = neural.ANN(16, 2, 3, 2)
output = ann.compute(np.array([3, 5, 9]))[-1]
```
The `neural.ANN.backpropagate` method handles the supervised learning of the network. It takes 3 parameters (and 1 optional), input-target pair list, number of gradient descent iterations to be performed, learning rate (gradient descent eta), and, optionally, output level (this can be either `True` or `False`).

In `props.py`, an example input-target pair list can be found, `props.eXample`. This can be used as an example training set for a neural network with 9 inputs and 2 outputs. The code below will perform the gradient descent algorithm 3 million times with a learning rate of 0.3, using samples defined in `props.eXample`:
```py
import neural
import props

ann = neural.ANN(16, 2, 9, 2)
ann.backpropagate(props.eXample, 3000000, 0.3)
```
