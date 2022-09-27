import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# from global_variables import spiral_data
# np.random.seed(0)

# sets random seed & default data type for numpy to use
# dot product for numpy sometimes uses a different datatype
nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # inputs first so we don't need to transpose weights on the forward pass
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # the first input to np.zeros IS the shape, therefore we use double brackets
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:  # ReLU --> Rectified Linear Unit
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        # subtracting max to avoid overflow error
        # axis = 1 adds rows
        # keepdims=True retains dimensions of original matrix
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    # output = output from model
    # y = intended target values
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    # y_prediction = values from NN
    # y_true = target training values
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # if scalar values e.g. [1, 0]
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # if one-hot encoded e.g. [[0,1],[1,0]]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# we are considering this the output layer, we defined 3 classes, therefore 3 neurons
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# note that we've initialized data randomly,
# as such we get a near perfect split of probability among the 3 outputs (neurons)
print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print('Loss: ',loss)