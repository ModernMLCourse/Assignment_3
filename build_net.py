"""
You should create a container class to host your neural network. The
class should contain all the layers in the order they are designed to
operate, and it should have a method to feed forward the input data
and return the network prediction.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


class Sequential():
    def __init__(self,layer_list):
        """
        This class hosts the layers of a neural network in the order they
        are intended to operate.
        :param layer_list: a python list of all layers making up the designed
        neural network
        """
        self.layers = # TODO: store a list of ordered network layers

    def forward(self,x):
        """
        A function passing the input through all layers and returning the
        network prediction
        :param x: an input of dimensions number of samples (batch size) x number of elements
        in an input vector (dimension of input vector)
        :return: network prediction
        """
        #TODO: implement a feed forward function to pass the input
        # x through all layers in "self.layers"
        return y