import numpy as np
from functions import spiral_data

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]
X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # if you experience a "dead network" i.e. lots of zeros, change the biases to a non-zero value
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Rectified Linear Unit
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# we're creating 2D data, so only 2 features (x & y)
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)

activation1.forward(layer1.output)
print(activation1.output)
