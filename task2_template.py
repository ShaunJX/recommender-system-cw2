import torch
from torch import nn
from torch.optim import SGD
import math
import sys
import os
import csv
import numpy as np
import sqlite3

def trainModel():
    # Define linear regression model
    print("linear regression")

    # Hyperparameters

    # Training Data

    # Ground truth Data

    # Initialize model

    # Define loss function, for this example we will be using Mean Squared Error

    # Define optimizer (Stochastic Gradient Descent) with a learning rate of 0.01

    # Training iteration 

        #initialize total loss variable

        # nested for loop to implement Stochastic Gradient Descent
            
            # Forward pass

            #get loss by comparing predicted value against the ground truth
            
            # Clear gradients

            # Back propagation (to compute gradients)

            # Update parameters          
            
            # Accumulate loss
        
        #print progress every 100 epochs

    # inferencing

    #Display results

if __name__ == "__main__":
    trainModel()