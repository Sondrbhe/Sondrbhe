# -*- coding: utf-8 -*-
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

sizedata = 1200
#V_in = spio.loadmat('Vin4', squeeze_me=True) 
#ESP_in = spio.loadmat('ESPinputV12', squeeze_me=True) 
#ESP_out = spio.loadmat('ESPoutputV12', squeeze_me=True) 
#x_train = V_in['Vin']
#y_train = V_out['Vcc']
#
#x_train = ESP_in['input']
#y_train = ESP_out['output']

data = spio.loadmat('ESPV12.mat', squeeze_me=True)
x_train = data['u']
y_train = data['y']

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


#print(y_train)
#print(y_train[1,0])

for i in range(len(y_train)):
    y_train[i,0] = (y_train[i,0] - min_y1)/(maks_y1 - min_y1) 
    x_train[i,0] = (x_train[i,0] - min_x1)/(maks_x1 - min_x1) 
    y_train[i,1] = (y_train[i,1] - min_y2)/(maks_y2 - min_y2) 
    x_train[i,1] = (x_train[i,1] - min_x2)/(maks_x2 - min_x2)    
    y_train[i,2] = (y_train[i,2] - min_y3)/(maks_y3 - min_y3)

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


data = [[], zip(x, y)]

# construct individual nodes
reservoir = Oger.nodes.ReservoirNode(output_dim=10)
readout = Oger.nodes.RidgeRegressionNode()

# build network with MDP framework
flow = mdp.Flow([reservoir, readout])


# Nested dictionary
gridsearch_parameters = {reservoir:{'input_scaling': mdp.numx.arange(0.07,0.08,0.01), 'spectral_radius': mdp.numx.arange(0.99,1,0.01), 'leak_rate': mdp.numx.arange(0,0.2,0.1),'_instance':range(5)}}


# Instantiate an optimizer
opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
#error = opt.mean_and_var()
# Do the grid search
#opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5)
opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.leave_one_out)


#opt.plot_results([(reservoir, '_instance')])  # OBS: has to hack Oger toolbox, for this to work
# Get the optimal flow and run cross-validation with it 
opt_flow = opt.get_optimal_flow(verbose=True)


print 'Performing cross-validation with the optimal flow. Note that this error can be slightly different from the one reported above due to another division of the dataset. It should be more or less the same though.'

errors = Oger.evaluation.validate(data, opt_flow, Oger.utils.nrmse, cross_validate_function=Oger.evaluation.leave_one_out)
#print 'Mean error over folds: ' + str(sp.mean(errors))



