import math
import numpy as np
layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)

print(exp_values)

norm_values = []

for value in exp_values:
    norm_values.append(value / sum(exp_values))

print(norm_values)
print(sum(norm_values))
