# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:37:33 2020

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

esn_4tanks = RNN.EchoStateNetwork(neu = 800,n_in = 2,n_out = 3,gama = 0.1,ro = 0.99,psi = 0.0,in_scale = 0.1,bias_scale = 0.1, initial_filename = "4tanks1", load_initial = False, save_initial = False,output_feedback = False)

dirpath = os.getcwd()
print("current directory is : " + dirpath)
data = io.loadmat('ESPV30.mat')

y_data = data['y']
y_max1 = max(y_data[:,0])
y_max2 = max(y_data[:,1])
y_max3 = max(y_data[:,2])
y_min1 = min(y_data[:,0])
y_min2 = min(y_data[:,1])
y_min3 = min(y_data[:,2])

y_max = np.array([y_max1, y_max2, y_max3])
y_min = np.array([y_min1,y_min2,y_min3])
y_data_normal = feature_scaling(y_data,y_max,y_min)


u_data = data['u']
u_max1 = max(u_data[:,0])
u_max2 = max(u_data[:,1])
u_min1 = min(u_data[:,0])
u_min2 = min(u_data[:,0])
u_max = np.array([u_max1,u_max2])
u_min = np.array([u_min1,u_min2])
u_data_normal = feature_scaling(u_data,u_max,u_min)


y_data_training = y_data_normal[:13500,:]
u_data_training = u_data_normal[:13500,:]

y_data_val = y_data_normal[13500:,:]	 
u_data_val = u_data_normal[13500:,:]

simtime = u_data.shape[0]
simtime1 = u_data_val.shape[0]
regularization = 1e-8	 
drop = 200

esn_4tanks.offline_training(u_data_training,y_data_training,regularization,drop)
 
 	 
esn_4tanks.reset()
y_pred = np.empty_like(y_data)

for k in range(simtime1):
    y_pred[k+13500] = esn_4tanks.update(u_data_normal[k]).flatten()


plt.figure()
plt.subplot(311)
plt.plot(y_pred[13500:,0],color='red', label='$p_{bh}$')
plt.plot(y_data_normal[13500:,0],color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')


plt.subplot(312)
plt.plot(y_pred[13500:,1],color='red',label='$p_{wh}$')
plt.plot(y_data_normal[13500:,1],color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(313)
plt.plot(y_pred[13500:,2],color='red',label='$q$')
plt.plot(y_data_normal[13500:,2],color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')




error = np.abs(y_pred - y_data_normal)
 
training_error = np.mean(error[200:13500],axis = 0)

val_error = np.mean(error[13500:], axis = 0)

print("This training gave the error: ", training_error, " and validation error: ",  val_error)

pickle_file = open('esn4tanks.pickle','wb')

pickle.dump(esn_4tanks,pickle_file)

pickle_file.close()
 
 
