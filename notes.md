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

