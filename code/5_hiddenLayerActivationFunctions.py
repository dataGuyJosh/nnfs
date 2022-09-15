import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# from global_variables import spiral_data
# np.random.seed(0)

# sets random seed & default data type for numpy to use
# dot product for numpy sometimes uses a different datatype
nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

# # rectified linear activation function
# for i in inputs:
#     output.append(max(0, i))
# print(output)


X, y = spiral_data(100, 3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # inputs first so we don't need to transpose weights on the forward pass
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # the first input to np.zeros IS the shape, therefore we use double brackets
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:  # ReLU --> Rectified Linear Unit
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# 2D data for XY plot, therefore 2 inputs (5 neurons is arbitrary)
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer1.forward(X)
# note we still have negative values
# these should be changed to 0 after the activation function!
print(layer1.output)
# now we have 0 values! note that if we were seeing too many we could initialize biases to a non-zero value
activation1.forward(layer1.output)
print(activation1.output)