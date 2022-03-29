"""
EE468: Neural Networks and Deep Learning class.
Third assignment-- Neural Networks
===============================================
Instructor notes:
-----------------
This script is meant as a guid for students to learn Python and do
Assignment 3 of the course. It introduces students to python classes
for the purpose of developing complete neural networks. NOTE: Only the
construction of a network is discussed here along with the implementation
of the forward pass. Backward pass and network training is deferred to
another assignment.

Happy coding... ;-)
============================
Author: Muhammad Alrabeiah
Date: Mar. 2022
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from build_net import Sequential
from layers import * # Read about the asterisk sign and its role with import


# Load some data
D = #TODO: load dataset from .mat file
x_trn = #TODO: read training inputs
y_trn = #TODO: read training labels
x_val = #TODO: read validation inputs
y_val = #TODO: read validation labels

#TODO: Plot training and validation data in two separate scatter plots
# Example: plt.scatter(x_trn[:, 0], x_trn[:, 1], c=y_trn, s=40, cmap=plt.cm.Spectral)

# Build a neural network
# TODO: use the classes you coded in script "layers.py" to create objects for
#  each layer of the network
fc_1 = #TODO: first layer of linear combiners
relu_1 = #TODO: first layer of activation functions
fc_2 = #TODO: second layer of linear combiners
relu_2 = #TODO: second layer of activation functions
fc_3 = #TODO: third layer of linear combiners
sfm = #TODO: softmax layer

layer_list = #TODO: create an ORDERED list of layers

net = #TODO: use Sequential class to construct an object representing your network

# Load trained parameters
#TODO: here you should do the following:
# 1) Load the trained parameters from the file "trained_parameters.mat".
# 2) Assign those parameters to their corresponding weights and biases in the network object "net".
# Hint: make sure the dimensions of the weights and biases are correct.

# Test network on validation dataset
#TODO: Test the network you have created on the validation data by doing
# the following:
# (1) Compute network prediction accuracy (See assignment for the equation)
# and print it on the screen using "print('Accuracy is {}'.format())". HINT:
# Read about using "format" with python print function
# (2) Record all predictions and plot a scatter plot of x_val where the data
# are colored based on their "predicted" labels not the groundtruth labels,
# i.e., "y_val". HINT: Scatter plot should be similar to that in Line 37