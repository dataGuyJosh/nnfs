# ONE NEURON
inputs = [1.0, 2.0, 3.0, 2.5]
# one weight per input
weights = [0.2, 0.8, -0.5, 1.0]
# one bias per neuron
bias = 2.0
output = 0.0

for i, w in zip(inputs, weights):
    output += i * w

# print(output + bias)

# THREE NEURONS
inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

layer_outputs = []
for wts, b in zip(weights, biases):
    result = 0
    for i, w in zip(inputs, wts):
        result += i * w
    layer_outputs.append(result + b)

print(layer_outputs)
