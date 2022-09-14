import numpy as np
np.random.seed(0)
# # Shape: (3, 4)
# inputs = [[1, 2, 3, 2.5],
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]]
# # Shape: (3, 4)
# weights1 = [[0.2, 0.8, -0.5, 1.0],
#             [0.5, -0.91, 0.26, -0.5],
#             [-0.26, -0.27, 0.17, 0.87]]
# biases1 = [2, 3, 0.5]

# weights2 = [[0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],
#             [-0.44, 0.73, -0.13]]
# biases2 = [-1, 2, -0.5]

# '''
# dot product of 2 matrices multiplies rows by columns,
# but we have less rows in inputs than columns in weights!
# so we need to transpose one of these matrices in order to perform a dot product

# recall that size at index 1 of element 1 in the dot product
# must match size at index 0 of element 2
# inputs[1] is of size 4 & weights[0] is of size 3
# so we need to swap rows & columns (transpose)
# '''
# layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
# print(layer1_outputs)

# # the inputs of layer 2 are the outputs of layer 1
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
# print(layer2_outputs)

# Shape (3, 4)
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# generally speaking
# initialise weights between 1 & -1
# initialise biases as 0


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # inputs first so we don't need to transpose weights on the forward pass
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # the first input to np.zeros IS the shape, therefore we use double brackets
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


num_inputs = len(X[0]) # 4
print(num_inputs)
layer1 = Layer_Dense(num_inputs, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)