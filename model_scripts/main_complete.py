import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from build_net_complete import Sequential
from layers_complete import *


# Load some data
D = sio.loadmat('assig3_dataset.mat')
x_trn = D['trn_inp']
y_trn = D['trn_labels']
x_val = D['val_inp']
y_val = D['val_labels']

plt.figure(0)
plt.scatter(x_trn[:, 0], x_trn[:, 1], c=y_trn, s=40, cmap=plt.cm.Spectral)
plt.grid(True)
plt.figure(1)
plt.scatter(x_val[:, 0], x_val[:, 1], c=y_val, s=40, cmap=plt.cm.Spectral)
plt.grid(True)

plt.show()

# Build a neural network
fc_1 = Linear(2,100)
relu_1 = ReLU(100)
fc_2 = Linear(100, 20)
relu_2 = ReLU(20)
fc_3 = Linear(20,3)
sfm = softmax(3)

layer_list = [fc_1,
              relu_1,
              fc_2,
              relu_2,
              fc_3,
              sfm]

net = Sequential(layer_list)

# Load trained parameters
D = sio.loadmat('trained_parameters.mat')
w_1, b_1 = D['w_1'], D['b_1']
w_2, b_2 = D['w_2'], D['b_2']
w_3, b_3 = D['w_3'], D['b_3']

w = net.layers[0].weights
print('Before')
print(w[:,:5])

net.layers[0].weights = w_1.T
net.layers[0].biases = b_1
net.layers[2].weights = w_2.T
net.layers[2].biases = b_2
net.layers[4].weights = w_3.T
net.layers[4].biases = b_3

w = net.layers[0].weights
print('After')
print(w[:,:5])

# Test network on validation dataset
acc = 0
pred_labels = []
for idx in range(x_val.shape[0]):
    # x = x_val[[idx],:]
    y = y_val[0,idx]
    prob = net.forward(x_val)
    pred = prob.argmax()
    pred_labels.append(pred)
    acc += (pred == y).astype(np.float32)

print('Accuracy is {0:4.3f}'.format(acc/x_val.shape[0]))
plt.figure(2)
plt.scatter(x_val[:, 0], x_val[:, 1], c=y_val, s=40, cmap=plt.cm.Spectral)
plt.grid(True)
plt.figure(3)
plt.scatter(x_val[:, 0], x_val[:, 1], c=pred_labels, s=40, cmap=plt.cm.Spectral)
plt.grid(True)
plt.show()