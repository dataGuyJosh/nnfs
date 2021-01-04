import numpy as np

layer_outputs = [4.8, 1.21, 2.385]
# exponentiation
# y=e^x --> 0.8952826639572619=e^4.8
exp_values = np.exp(layer_outputs)
# normalise values between 0 & 1
norm_values = exp_values / np.sum(exp_values)
# print(exp_values)
# print(norm_values)
# print(sum(norm_values))

# now let's this work with a "batch" of outputs

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

# axis=1 on a 2D matrix is the sum of each row i.e. we want this rather than a single number
# keepdims=True maintains each number as a single item array within the parent array
# (otherwise it just returns an array with 3 numbers)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)