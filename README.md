# Assignment 3

It introduces students to python classes
for the purpose of developing complete neural networks.

# Instructions

You are not expected to develop and train a neural network in this assignment, for this has already been done for you.
You are required to learn the designed neural network and implement it by completing the scripts provided in this repo.
You are expected to do the following:

1) Develop separate Python classes for each layer of the designed neural network. To do so, you need to:
   1) Understand the designed network.
   2) Understand the functions implement in each layer, i.e., linear combiner, ReLU,...etc.
   3) Use the class templates in "layers.py" to develop the layers.
2) Construct a Python class that hosts the neural network, called network class. To do so, you need:
   1) Understand how layers are stacked and connected in a neural network.
   2) Use the class template named Sequential in the file "build_net.py" to construct you network class.
3) Load the trained parameters in the file "trained_parameters.mat" to your network, and do the folloowing:
   1) Compute the accuracy of the model on the validation dataset.
   2) Get the predictions of the model and create a scatter plot of the validation input data where they are colored based on their predicted labels.

# Dataset Structure

The repo contains the spiral dataset generated for this assignment, file "assig3_dataset.mat". The structure of this dataset is as follows:
1) trn_inp: is a 600x2 training matrix, where each row is a vector of observed variables.
2) trn_labels: is a 1x600 vector of grounddtruth training labels, one per input vector. There are a total pf 3 classes, encoded as 0, 1, and 2.
3) val_inp: is a 300x2 validation matrix.
4) val_labels is a 1x300 vector of groundtruth validation labels 
