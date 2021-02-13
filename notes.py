# the dot product of 2 vectors results in a scalar (single) value
# dot product of a & b
a = [1, 2, 3]
b = [2, 3, 4]
dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
# print(dot_product)

'''
Part 5
Activation Functions
linear (y=x) --> a line, therefore we can only fit linear data
sigmoidal --> normalize x between 0 and 1, higher accuracy but typically slow
rectified linear (if x < 0, 0 else x) --> faster than sigmoidal, more accurate than linear!
'''

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

'''
Part 6
rectified linear is fast, but loses meaning for numbers less than 0 (as they are all 0)
how can we make negative values positive without removing the meaning of each number?
we can't (for example) just make all values absolute as -9 would become 9
i.e. we lose the ability to differentiate
using euler's number we can guarantee all numbers are positive without losing meaning
y = e^x
in theory we don't need to use euler's number but it will help later on
once we do this, we can normalise the values between 0 & 1
this process of exponentiation and normalisation is called softmax!
Input --> Exponentiate --> Normalize --> Output

an issue with exponentiation is that it quickly produces huge values
a simple solution is to subtract the largest value in each layer from every value in that layer
e.g. 1 2 3 would become -2 -1 0
because the largest value is 0, all values can only ever be between 0 & 1
this is identical once normalised so we're not breaking anything by doing this!
'''

'''
Part 7 - Calculating Loss with Categorical Cross-Entropy
Loss function is more useful than accuracy
Types of loss functions - Mean Absolute Error
- take the average difference between values calculated and target values
- used with regression
- "how" wrong is a model?
In general, the loss function of choice for classification (where we use softmax on the output layer) is Categorical Cross-Entropy
- the negative sum of the target value multiplied by the log of the predicted value for each value in the distribution
- formula simplifies to being the negative log of the target classes' predicted value (due to "One-Hot Coding")
There are many ways to calculate loss, it just so happens that this is convenient (for back propagation & optimization), popular & successful
One-hot encoding
You have a vector of n classes with zeros in all indexes other than the target label.
The Categorical Cross-Entropy formula is simplified as described above due to the fact that all indexes except for one are given a value of 0.
It just becomes negative log, technically still categorical cross entropy.
 
Logarithims
- natural log is base 'e'
'''

# Solving for x --> e ** x = b
import numpy as np
import math
b = 5.2
# print(np.log(b))
# print(math.e ** np.log(b))

