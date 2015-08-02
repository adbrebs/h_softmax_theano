import numpy as np
np.random.seed(42)

import theano
from theano import tensor, shared
floatX = theano.config.floatX

from h_softmax import h_softmax_cpu, h_softmax_gpu


#############
# Config
#############
gpu = False
input_size = 100
batch_size = 50
hidden_sizes = [10, 10]
h_softmax_level1_size = 10
h_softmax_level2_size = 100
output_size = h_softmax_level1_size * h_softmax_level2_size
learning_rate = 0.001


#############
# Initialize shared variables
#############
weight_matrices = []
buff = [input_size] + hidden_sizes
for i, j in zip(buff, buff[1:]):
    weight_matrices.append(
        shared(np.random.normal(size=(i, j)).astype(floatX)))

# First level of h_softmax
W1 = np.asarray(np.random.normal(
    size=(hidden_sizes[-1], h_softmax_level1_size)), dtype=floatX)
W1 = shared(W1)
b1 = shared(np.asarray(np.zeros((h_softmax_level1_size,)), dtype=floatX))

# Second level of h_softmax
W2 = np.asarray(np.random.normal(
    size=(h_softmax_level1_size, hidden_sizes[-1], h_softmax_level2_size)),
    dtype=floatX)
W2 = shared(W2)
b2 = shared(
    np.asarray(np.zeros((h_softmax_level1_size, h_softmax_level2_size)),
               dtype=floatX))


#############
# Build graph
#############
x = tensor.matrix('x')
y = tensor.ivector('y')
y_hat = x
for W in weight_matrices:
    y_hat = tensor.dot(y_hat, W)

if gpu:
    h_softmax = h_softmax_gpu
else:
    h_softmax = h_softmax_cpu

# This only computes the output corresponding to the target
y_hat_tg = h_softmax(W1, b1, W2, b2, y_hat, output_size, h_softmax_level1_size,
                     h_softmax_level2_size, batch_size, target=y)

# This computes all the outputs
y_hat_all = h_softmax(W1, b1, W2, b2, y_hat, output_size,
                      h_softmax_level1_size, h_softmax_level2_size, batch_size)

cost_training = -tensor.mean(tensor.log(y_hat_tg))

parameters = weight_matrices + [W1, b1, W2, b2]
grads = theano.grad(cost_training, parameters)
updates = [(w, w - learning_rate*grad) for (w, grad) in zip(parameters, grads)]


#############
# Compile functions
#############
training_fun = theano.function([x, y], [cost_training], updates=updates)
output_fun = theano.function([x], y_hat_all)


#############
# Test
#############
x_mat = 0.001*np.random.normal(size=(batch_size, input_size)).astype(floatX)
y_mat = np.ones(batch_size, dtype='int32')
print training_fun(x_mat, y_mat)
print output_fun(x_mat)
