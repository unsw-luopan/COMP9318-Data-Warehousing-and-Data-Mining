import numpy as np
import pandas as pd


def logistic_regression(data, labels, weights, num_epochs, learning_rate):  # do not change the heading of the function
    data = np.mat(data)
    labels=np.mat(labels)
    one = np.ones((data.shape[0],1))
    data = np.concatenate((one,data),axis=1)
    #print(np.shape(data))  (6000，3)
    weights = np.zeros((weights.shape[0],1))
    #print(np.shape(weights))  (3,1)
    #print(np.shape(labels)) (1,6000)
    for i in range(num_epochs):
        y = sigmoid(data*weights)
        error = labels.T-y
        weights = weights + learning_rate * data.T * error
    return weights

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
