# URLs
- [NNFS Youtube Playlist](https://youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
- [CS231 Course](https://cs231n.github.io/neural-networks-case-study/)

# Part 1 - Intro and Neuron Code
## Neural Network Calculations
- forward pass --> inputs * weights + biases
- structure
  - neurons
  - layers
    - input layer
    - hidden layers
    - output layer
- every neuron is connected to every other neuron in adjacent layers

# Part 2 - Coding a Layer
What is the input layer? There are usually values being tracked e.g. if predicting equipment failure, input values might be temperature, humidity and so on from sensors.

# Part 3 - The Dot Product
More on Weights & Biases
- used to tune our parameters
- weights impact output differently to biases

Understanding shape
At each dimension, what is the the size?
list_of_lists = [[1, 2, 3], [4, 5, 6]]
The shape of the 2D array above is (2, 3)

A tensor is an object which CAN be represented as an array, it is not equivalent though.

# Part 4 - Batches, Layers and Objects
Why do we show samples to a machine in batches?
- GPUs can be used to perform simple calculations in parallel
- helps with generalisation i.e. rather than showing a machine one example at a time, show it multiple

Higher batch sizes makes it easier for a neural network to fit data, however too high a batch size can lead to overfitting!

# Part 5 - Hidden Layer Activation Functions
Example activation functions
- (Unit) Step Function
  - output = input > 0 ? 1 : 0
  - output is always 0 or 1
  - doesn't tell us "how close" we were to 0 or 1
- Sigmoid
  - y = 1/(1+e^-x)
  - reliable due to granularity
  - x = input * weight + bias --> y = 1/(1 + e ^ -x)
  - suffers from the "vanishing gradient problem"
- Rectified Linear
  - output = input > 0 ? input : 0
  - y = ReLU(x)
  - fast to calculate compare to sigmoid
  - "it just works"

Generally the output layer has a different activation function to the hidden layers.

Why is rectified linear such a good activation function? It's non-linear AND fast!

Why are more neurons per layer useful? They provide additional "areas of effect" to shape the model to fit data (similar to segments on a 2D graph).

# Part 6 - Softmax Activation
Why not just stick with the rectified linear activation function?
- the first step in training a model is "how wrong is this model?" --> accuracy is not a good indication of that
```python
# if we're just "predicting", then the most accurate value is the largest
# output 1 is the largest and therefore most accurate
layer_outputs = [4.8, 1.21, 2.385]

# output 1 is the largest BUT output 2 & 3 are much closer than in the previous set
layer_outputs = [4.8, 4.79, 4.25]
```

In the example above, we can see that the bottom set of outputs are more desirable even though both models have equivalent accuracy (as their max values are both 4.8). 

The rectified Linear activation function 
- is exclusive in that there is no relation between neurons involved. 
- there also isn't any "bounding" i.e. no upper limit on values
- clips any negative values, which can potentially be a large portion of the dataset

What is our end objective?
- we want the output values to be a "probability distribution"
  - neuron to neuron values are normalized
  - we can measure "how right/wrong" a prediction is

Exponential Function
- y = e ^ x
- solves the negativity issue without losing the meaning of negative values i.e. 1 != -1

Softmax
- Softmax is exponentiation & normalization

Input | Exponentiate | Normalize | Output
-|-|-|-
[1, 2, 3] | [e^1, e^2, e^3] | [ (e^1)/(e^1+e^2+e^3), <br> (e^2)/(e^1+e^2+e^3), <br> (e^3)/(e^1+e^2+e^3) ] | [0.09, 0.24, 0.67]

An issue with exponentiation is that values become massive quickly, a solution to this is to subtract the largest value prior to exponentiation making all values negative or 0. After exponentiation, we normalize our range between 0 & 1.

# Part 7 - Calculating Loss with Categorical Cross-Entropy
In order to do backpropagation and optimization, we need to have some measure of how "wrong" the model is (a metric of error). For this, we use a loss function. In our case, with a softmax classifier, we'll be using categorical cross-entropy.

A model will output a probability distribution (confidence score), this means we have more information than just "accuracy" (whether a prediction was/was not correct) to train the model.

Categorical Cross-Entropy: the negative sum of the target value multiplied by the log of the predicted value for each value in the distribution. This simplifies to the negative log of the target classes predicted value (due to one-hot encoding).

L_i = -log(yi,k)
- L_i >> sample loss value
- i >> i-th same in a set
- k >> target label index, index of correct class probability
- y >> predicted values


One-hot Encoding: a vector of n-classes, where the index of the target class is 1, while all other indexes are of value 0.
```
Classes: 2
Label: 1
One-hot: [0,1]

Classes: 3
Label: 0
One-hot: [1,0,0]
```

Logarithms
- Natural Log: y = log_e * x = ln(x)
- e ~ 2.718


# Part 8 - Implementing Loss
We've found a problem! The negative log of 0 is infinity, technically this just means an output is infinitely wrong (which is correct) BUT it becomes a problem when we want to calculate the batch loss by taking a mean. A solution to this is to "clip" the values by a very small amount (e.g. 1e-7) in order to avoid dealing with infinity. This makes our new range 1e-y to 1 - 1e-7.

How do we decrease loss? We update the weights and biases! More info in later sections.