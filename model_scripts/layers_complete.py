"""
This script creates three important classes each of which could be viewed as a layer
Before completing the script, please carefully read the notes provided with each class
as well as the assignment sheet.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#########################

class Linear():
    def __init__(self,
                 in_dim,
                 out_dim,
                 use_bias=True,
                 initializer='xavier'):
        """
        A class implementing the linear combiner of a single layer
        NOTE: the weights and biases of this layer are of the shape
        (number of neurons, dimension of input vector) and (number of neurons,),
        respectively
        :param in_dim: size of the input vector (i.e., number of elements in the input vector)
        :param out_dim: number of neurons
        :param use_bias: True--> implements bias addition. False--> no bias implemented
        :param initializer: Type of weight initialization
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.initializer = initializer

        # Define the weights and biases of the linear layer
        if self.initializer == 'xavier':
            self.weights = (2*np.random.rand(self.in_dim,self.out_dim) - 1)*(np.sqrt(6)/np.sqrt(self.in_dim+self.out_dim))
        elif  self.initializer == 'normal':
            self.weights = np.random.randn(self.in_dim, self.out_dim)

        if self.use_bias:
            self.biases = np.zeros((self.out_dim,))

    def forward(self, x):
        """
        forward pass through a linear combiner layer
        :param x: dimensions are number of samples (batch size) x number of elements
        in an input vector (dimension of input vector)
        :return: linear combiner output
        """
        y = np.matmul(x, self.weights) + self.biases if self.use_bias else np.matmul(x, self.weights)
        return y

    def backward(self,dLdy):
        ValueError('Not implemented yet!')

#########################

class ReLU():
    def __init__(self,
                 dim):
        """
        A class implementing element-wise relu function
        :param dim:
        """
        self.num_neurons = dim

    def forward(self,x):
        """
        forward pass through a ReLU layer
        :param x: dimensions are number of samples (batch size) x input dimension
        :return: linear combiner output
        """
        y = np.maximum(x,0)
        return y

    def backward(self,dLdy):
        ValueError('Not implemented yet!')

#########################

class softmax():
    def __init__(self, dim):
        """
        A class implementing a stable version of softmax
        :param dim:
        """
        self.dim = dim

    def forward(self,x):
        constant = np.max(x,axis=1)
        x = x - constant
        nume = np.exp(x)
        deno = np.sum(np.exp(x), axis=1)
        y = nume/deno
        return y