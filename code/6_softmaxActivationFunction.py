# import numpy as np
# import nnfs

# nnfs.init()


# layer_outputs = [[4.8, 1.21, 2.385],
#                  [8.9, -1.81, 0.2],
#                  [1.41, 1.051, 0.026]]


# # now with numpy
# exp_values = np.exp(layer_outputs)

# # axis = 1 changes the behaviour of sum to add rows
# # instead of all values (giving us 3 results)
# # keepdims=True retains the dimensions of the original matrix
# # i.e. instead of [8.395 7.29 2.487] the result is
# # [[8.395]
# #  [7.29 ]
# #  [2.487]]
# norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# print(norm_values)

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# from global_variables import spiral_data
# np.random.seed(0)

# sets random seed & default data type for numpy to use
# dot product for numpy sometimes uses a different datatype
nnfs.init()


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


class Activation_Softmax:
    def forward(self, inputs):
        # subtracting max to avoid overflow error
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# we are considering this the output layer, we defined 3 classes, therefore 3 neurons
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# note that we've initialized data randomly, 
# as such we get a near perfect split of probability among the 3 outputs (neurons)
print(activation2.output[:5])
