# weights & biases are the "nobs" we tune to make a model conform to a set of data
# we're making them ourselves right now but they will ideally be automated down the line
# you can make a model without weights or biases but it's usually beneficial,
# it's almost never detrimental to have a bias

# a tensor is an object that "can" be represented as an array (not always)

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# use loops to clean up code/make it more scalable
layer_outputs = []  # output of current layer
for neuron_weights, neuro_bias in zip(weights, biases):
    neuron_output = 0  # output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuro_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

# let's do the same above using numpy to clean up our code even further

import numpy as np

# vector
inputs = [1, 2, 3, 2.5]
# matrix
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# simplify our code even further using numpy's dot product function
# numpy.dot returns based on the first argument i.e weights is a matrix so we get a vector back
# the alternative will give a shape error
output = np.dot(weights, inputs) + biases
print(output)
