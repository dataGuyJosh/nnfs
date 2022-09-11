inputs = [1.2, 5.1, 2.1]
# one weight per input
weights = [3.1, 2.1, 8.7]
# one bias per neuron
bias = 3
result = 0

for i, j in zip(inputs, weights):
    result += i * j
    
result += bias

print(result)
