import numpy as np

# Dot Product
a = [1, 2, 3]
b = [2, 3, 4]

result = 0
for i, j in zip(a, b):
    result += i * j
print(result)

# One Neuron
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

# weights first, inputs second
# inputs is a vector while weights is a matrix
# therefore order will change the result of dot product
output = np.dot(weights, inputs) + bias
# this will throw a "shapes" error!
# output = np.dot(inputs, weights) + bias
print(output)

# Three Neurons
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
output = np.dot(weights, inputs) + biases
print(output)
