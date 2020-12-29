import numpy as np

# remember each input can be equivalent to a "feature"
# e.g. properties of a server like local temperature, humidity etc...
inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

# in order to transpose the weights array (so that rows and columns match up for the dot product)
# inputs must have the same number of columns as weights number of rows
# we need to convert weights to a numpy array, then transpose it
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# feed the output for layer 1 into layer 2
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

# now let's do this with multiple layers
# set a seed for random so our output matches tutorial
np.random.seed(0)

# these input values can get out of hand in hidden layers, so let's scale them between -1 & 1
X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# arg1 input size (n_inputs)
# arg2 number of neurons (n_neurons)
# NOTE: n_inputs in layer2 must match n_neurons from layer1!!
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)