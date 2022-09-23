import math

softmax_output = [0.7, 0.1, 0.2]
# target_class = 0
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])
print(loss)
# note that index 1 & 2 equate to 0, therefor it's equivalent to
loss = -math.log(softmax_output[0])
print(loss)
# larger output values are considered better, and therefore have lower loss
print(
    -math.log(0.7),  # more confident, lower loss
    -math.log(0.5)   # less confident, higher loss
)