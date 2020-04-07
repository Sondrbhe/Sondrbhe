# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:42:05 2020

@author: iver_
"""

 
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
#import Project_esnmpc.RNN as RNN
import RNN
import pickle
import os

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
# leak rate (gama) = 0.14 e ait
esn_espV12 = RNN.EchoStateNetwork(neu = 100,n_in = 2,n_out = 3,gama = 0.14,ro = 0.99,psi = 0.0,in_scale = 0.07,bias_scale = 0.1, initial_filename = "4tanks1", load_initial = False, save_initial = False,output_feedback = False)

dirpath = os.getcwd()
print("current directory is : " + dirpath)
data = io.loadmat('ESPV12.mat')

y_data = data['y']
y_max = np.array([1.05e07,4.3e06, 0.018])
y_min = np.array([6.2e06,2e06,0.001])

y_data_normal = feature_scaling(y_data,y_max,y_min)


u_data = data['u']
u_max = np.array([1,65])
u_min = np.array([0.4,35])
u_data_normal = feature_scaling(u_data,u_max,u_min)

data_size = y_data.shape[0]
scaling = 0.9
limit = int(data_size*scaling)

y_data_training = y_data_normal[:limit,:]
u_data_training = u_data_normal[:limit,:]

y_data_val = y_data_normal[limit:,:]	 
u_data_val = u_data_normal[limit:,:]

simtime = u_data.shape[0]
simtime1 = u_data_val.shape[0]
regularization = 1e-8	 
drop = 200

esn_espV12.offline_training(u_data_training,y_data_training,regularization,drop)
 
 	 
# esn_4tanks.reset()
# y_pred = np.empty_like(y_data)

# for k in range(simtime):
#     y_pred[k] = esn_4tanks.update(u_data_normal[k]).flatten()
esn_espV12.reset()
y_pred = np.empty_like(y_data_val)

for k in range(simtime1):
    y_pred[k] = esn_espV12.update(u_data_val[k]).flatten()


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




error = np.abs(y_pred - y_data_normal[limit:])
 
training_error = np.mean(error[200:limit],axis = 0)

val_error = np.mean(error, axis = 0)

print("This training gave the error: ", training_error, " and validation error: ",  val_error)

pickle_file = open('esnesp.pickle','wb')

pickle.dump(esn_espV12,pickle_file)

pickle_file.close()
 
 
