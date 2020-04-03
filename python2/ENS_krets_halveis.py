'''
Author: Eric Aislan Antonelo
email: eric.antonelo@gmail.com
- Part of the Minicourse on Echo State Networks
Last updated on December 2017
'''

import Oger
import pylab
import mdp

from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, subplot, show
import numpy as np
import scipy.io as spio

""" This example demonstrates a very simple reservoir+readout setup on the 30th order NARMA task.
"""

V_in = spio.loadmat('Vin4', squeeze_me=True) 
V_out = spio.loadmat('Vcc4', squeeze_me=True) 

x_train = V_in['Vin']
y_train = V_out['Vcc']



maks = max(y_train)
mini = min(y_train)
maks_2 = max(x_train)
min_2 = min(x_train)

for i in range(len(y_train)):
    y_train[i] = (y_train[i] - mini)/(maks - mini) 
    x_train[i] = (x_train[i] - min_2)/(maks_2 - min_2) 

#Dataset parameters
size = 10000
chunks = 10
column = 1
###################

x = np.zeros((chunks, size,column))
y = np.zeros((chunks, size,column))


for i in range(chunks):    
    for j in range(size):
        x[i,j] = x_train[j+(i*size)]
        y[i,j] = y_train[j+(i*size)]


##  Get the dataset
#[x, y] = Oger.datasets.narma30()

## Construct reservoir
reservoir = Oger.nodes.ReservoirNode(output_dim=400, input_scaling=0.09, spectral_radius = 0.98)
#Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)

# build network with MDP framework, and train
readout = Oger.nodes.RidgeRegressionNode()

flow = mdp.Flow([reservoir, readout], verbose=1)

# - 9 samples are selected for training, and 1 sample for test
n = -1
data = [None, zip(x[0:n], y[0:n])]
sample = data[1][0]     # first sample

# train the flow 
flow.train(data)

# apply the trained flow to the training data and test data
trainout = flow(x[0])
testout = flow(x[-1])

# compute test error metric NRMSE
print "NRMSE: " + str(Oger.utils.nrmse(y[-1], testout))

nx = 4
ny = 1

#plot the input
pylab.subplot(nx, ny, 1)
pylab.plot(x[0])

#plot everything
pylab.subplot(nx, ny, 2)
pylab.plot(trainout, 'r')
pylab.plot(y[0], 'b')

pylab.subplot(nx, ny, 3)
pylab.plot(testout, 'r',label = "predicted")
pylab.plot(y[-1], 'b', label = "true")

pylab.subplot(nx, ny, 4)
pylab.plot(reservoir.inspect()[-1])
pylab.show()
