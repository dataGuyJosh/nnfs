inputs = [1.2, 5.1, 2.1]
# one weight per input
weights = [3.1, 2.1, 8.7]
# one bias per neuron
bias = 3
output = 0

for i, j in zip(inputs, weights):
    output += i * j

output += bias

print(output)
