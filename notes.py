# the dot product of 2 vectors results in a scalar (single) value
# dot product of a & b
a = [1, 2, 3]
b = [2, 3, 4]
dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
# print(dot_product)

# Part 5
# Activation Functions
# linear (y=x) --> a line, therefore we can only fit linear data
# sigmoidal --> normalize x between 0 and 1, higher accuracy but typically slow
# rectified linear (if x < 0, 0 else x) --> faster than sigmoidal, more accurate than linear!
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
    output.append(max(0, i))
# print(output)

# The Daniel Optimizer

# Review this at some point
# https://cs231n.github.io/neural-networks-case-study/
import numpy as np

np.random.seed(0)


def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


import matplotlib.pyplot as plt

X, y = spiral_data(100, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
# plt.show()
