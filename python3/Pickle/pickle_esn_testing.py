# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
#import Project_esnmpc.RNN as RNN
import pickle
import os

pickle_file = open('esnesp.pickle','rb')

ESP_in = io.loadmat('ESPinputV12', squeeze_me=True) 
ESP_out = io.loadmat('ESPoutputV12', squeeze_me=True) 

x_train = ESP_in['input']
y_train = ESP_out['output']

def feature_scaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = (x - xmin)/(xmax - xmin)
    else:
        y = x 	 
    return y
 

def feature_descaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = x*(xmax-xmin) + xmin 
    else:
        y = x
    return y

y_data = y_train

y_max = np.array([1.05e07,4.3e06, 0.018])
y_min = np.array([6.2e06,2e06,0.001])

y_data_normal = feature_scaling(y_data,y_max,y_min)


u_data = x_train 

u_max = np.array([1,65])
u_min = np.array([0.4,35])

u_data_normal = feature_scaling(u_data,u_max,u_min)


test_size = 1000

nettverk = pickle.load(pickle_file)

#filename = 'esnesp.pickle' 
y_pred = np.empty_like(y_data)
simtime = u_data.shape[0]
for k in range(simtime):
    y_pred[k] = nettverk.update(u_data_normal[k]).flatten()




plt.figure(2)
plt.subplot(311)
plt.plot(y_pred[200:test_size,0],color='red', label='$p_{bh}$ pred')
plt.plot(y_data_normal[200:test_size,0],color='blue', label='$p_{bh}$ real')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')


plt.subplot(312)
plt.plot(y_pred[200:test_size,1],color='red',label='$p_[bh]$ pred')
plt.plot(y_data_normal[200:test_size,1],color='blue', label='$p_{bh}$ real')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(313)
plt.plot(y_pred[200:test_size,2],color='red',label='$q$ pred')
plt.plot(y_data_normal[200:test_size,2],color='blue', label='$q$ real')
plt.grid()
plt.legend()
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')







