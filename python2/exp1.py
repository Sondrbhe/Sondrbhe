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


#V_in = spio.loadmat('Vin4', squeeze_me=True) 
ESP_in = spio.loadmat('ESPinputV1', squeeze_me=True) 
ESP_out = spio.loadmat('ESPoutputV1', squeeze_me=True) 
#x_train = V_in['Vin']
#y_train = V_out['Vcc']

x_train = ESP_in['input']
y_train = ESP_out['output']

print(y_train[:,0])
print(y_train.shape)
print(len(y_train[:,0]))
#
#
maks_y1 = max(y_train[:,0])
min_y1 = min(y_train[:,0])
maks_y2 = max(y_train[:,1])
min_y2 = min(y_train[:,1])
maks_y3 = max(y_train[:,2])
min_y3 = min(y_train[:,2])

maks_x1 = max(x_train[:,0])
min_x1 = min(x_train[:,0])
maks_x2 = max(x_train[:,1])
min_x2 = min(x_train[:,1])


print(y_train)
print(y_train[1,0])

for i in range(len(y_train)):
    y_train[i,0] = (y_train[i,0] - min_y1)/(maks_y1 - min_y1) 
    x_train[i,0] = (x_train[i,0] - min_x1)/(maks_x1 - min_x1) 
    y_train[i,1] = (y_train[i,1] - min_y2)/(maks_y2 - min_y2) 
    x_train[i,1] = (x_train[i,1] - min_x2)/(maks_x2 - min_x2)    
    y_train[i,2] = (y_train[i,2] - min_y3)/(maks_y3 - min_y3)


sizedata = 600

x = np.zeros((10, sizedata,2))
y = np.zeros((10, sizedata,3))

k = 0
i = 0

for i in range(10):
    for k in range(3): 
        for j in range(sizedata):
            y[i,j,k] = y_train[j+(i*sizedata),k]
            
###############
            
for i in range(10):
    for k in range(2):
        for j in range(sizedata):
            x[i,j,k] = x_train[j+(i*sizedata),k]




            
print(x[9])

print(x_train)

#for k in range(3):
#    print(k)    
#    for j in range(100):
##        x[i,j,k] = x_train[j,k]
#        y[i,j,k] = y_train[j,k]

#
#for i in range(10):    
#    for j in range(10000):
#        x[i,j] = x_train[j+(i*10000)]
#        y[i,j] = y_train[j+(i*10000)]

##  Get the dataset
#[x, y] = Oger.datasets.narma30()

## Construct reservoir
reservoir = Oger.nodes.ReservoirNode(output_dim=100, input_scaling=0.07, spectral_radius = 1)
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

unorm_1 = (maks_y1 - min_y1)/10**5
unorm_2 = (maks_y2 - min_y2)/10**5
unorm_3 = (maks_y3 - min_y3)*3600

plt.figure(1)
plt.plot(y[9,:,0]*(unorm_1) , 'b')
plt.plot(testout[:,0]*(unorm_1) , 'r', label='$p_{bh}$')
plt.ylabel('$p_{bh} \ [bar]$')
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(y[9,:,1]*(unorm_2) , 'b')
plt.plot(testout[:,1]*(unorm_2) , 'r',label='$p_{wh}$')
plt.ylabel('$p_{wh} \ [bar]$')
plt.legend()
plt.grid()


plt.figure(3)
plt.plot(y[9,:,2]*(unorm_3) , 'b')
plt.plot(testout[:,2]*(unorm_3) , 'r',label='$q}$')
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')
plt.legend()
plt.grid()
#
#
##plot the input
#pylab.subplot(nx, ny, 1)
#pylab.plot(x[0])
#
##plot everything
#pylab.subplot(nx, ny, 2)
#pylab.plot(trainout, 'r')
#pylab.plot(y[0,:,0], 'b')
#
#
#pylab.subplot(nx, ny, 3)
#pylab.plot(testout*(maks_2 - min_2) , 'r',label = "predicted")
#pylab.plot(y[-1]*(maks - mini) , 'b', label = "true")
#
#pylab.subplot(nx, ny, 4)
#pylab.plot(reservoir.inspect()[-1])
#pylab.show()
