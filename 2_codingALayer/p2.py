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
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2.0
bias2 = 3.0
bias3 = 0.5

for wts, b in zip([weights1, weights2, weights3], [bias1, bias2, bias3]):
    output = 0.0
    for i, w in zip(inputs, wts):
        output += i * w
    print(output + b)
