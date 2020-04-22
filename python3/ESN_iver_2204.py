# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:32:50 2020

@author: iver_
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
#import Project_esnmpc.RNN as RNN
import ESN_iver_bib as ESN
import pickle
import mdp

def normalize(x, max, min):
    y = (x-min)/(max-min)
    return y

data = io.loadmat('ESPV12.mat')

# Import input and output data
u_data = data['u']
y_data = data['y']

# Limits for inputdata, these values have a margin to the real min and max
y_max = np.array([1.05e07,4.3e06, 0.018])
y_min = np.array([6.2e06,2e06,0.001])
u_max = np.array([1,65])
u_min = np.array([0.4,35])

# Scaling (normalization)
y_data_normal = normalize(y_data,y_max,y_min)
u_data_normal = normalize(u_data,u_max,u_min)

# Scaling for splitting into training and test sets.
data_size = y_data.shape[0]
scaling = 0.9
limit = int(data_size*scaling)

# Training
y_data_training = y_data_normal[:limit,:]
u_data_training = u_data_normal[:limit,:]

# Validation / test
y_data_val = y_data_normal[limit:,:]	 
u_data_val = u_data_normal[limit:,:]

simtime = u_data.shape[0]
simtime1 = u_data_val.shape[0]
regularization = 1e-8	 
drop = 200

# Creating ESN object in ESN_bib
network = ESN.EchoStateNetwork(nodes = 100, input_size = 2, output_size = 3, leak_rate = 0.14,
                               spectral_radius = 0.99, psi = 0.0, input_scaling = 0.07, bias_scaling = 0.1,
                               output_feedback = False)


# Noise should go here (format dataset)
y_data_training1 = y_data_training + np.random.normal(0, 0.1, [10800, 3])
u_data_training1 = u_data_training + np.random.normal(0, 0.1, [10800, 2])


# Train network with training data
network.off_training(u_data_training1,y_data_training1,regularization,drop)

network.reset()
y_pred = np.empty_like(y_data_val)

# Uses the network to predict next state
for k in range(simtime1):
    y_pred[k] = network.update(u_data_val[k]).flatten()
    

# # Check for both training and test
# y_pred = np.empty_like(y_data_normal)
# for k in range(simtime):
#     y_pred[k] = network.update(u_data_normal[k]).flatten()
    
    

#########################################################################
# Plotting

plt.figure(1)
plt.plot(y_data_training[:,0],color='red', label='$p_{bh}$')
plt.plot(y_data_training1[:,0],color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')

plt.figure(2)
plt.subplot(311)
plt.plot(y_pred[200:,0],color='red', label='$p_{bh}$')
plt.plot(y_data_val[200:,0],color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')


plt.subplot(312)
plt.plot(y_pred[200:,1],color='red',label='$p_{wh}$')
plt.plot(y_data_val[200:,1],color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(313)
plt.plot(y_pred[200:,2],color='red',label='$q$')
plt.plot(y_data_val[200:,2],color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')

# plt.figure(2)
# plt.subplot(311)
# plt.plot(y_pred[200:,0],color='red', label='$p_{bh}$')
# plt.plot(y_data_normal[200:,0],color='blue', label='$p_{bh}$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{bh} \ [bar]$')


# plt.subplot(312)
# plt.plot(y_pred[200:,1],color='red',label='$p_{wh}$')
# plt.plot(y_data_normal[200:,1],color='blue', label='$p_{bh}$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{wh} \ [bar]$')

# plt.subplot(313)
# plt.plot(y_pred[200:,2],color='red',label='$q$')
# plt.plot(y_data_normal[200:,2],color='blue', label='$p_{bh}$')
# plt.grid()
# plt.legend()
# plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')

#########################################################################
# Error calculations
var = y_data_normal.std(ddof=1)**2

error = (y_data_normal[limit:] - y_pred)**2
nrmse = np.sqrt(error.mean() / var)

print("NRMSE: ", nrmse)

#########################################################################
# Saving the network to a pickle file. For exporting the trained network
# pickle_file = open('esnesp.pickle','wb')

# pickle.dump(esn_espV12,pickle_file)

# pickle_file.close()
