import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


class Sequential():
    def __init__(self,layer_list):

        self.layers = layer_list

    def forward(self,x):
        for idx, lay in enumerate(self.layers):
            x = lay.forward(x)
        return x