import math

softmax_output = [0.7, 0.1, 0.2]

target_class = 0
# with a target class of 0 our one-hot vector is as follows
target_output = [1, 0, 0]
# remember that most of our outputs are 0, so the following code
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])
print(loss)
# is equivalent to this
loss = -math.log(softmax_output[0])
print(loss)
# note that as confidence (softmax_output) increases
print(-math.log(0.5))
# loss decreases i.e. confidence is inversely proportionate to loss
print(-math.log(0.7))
