'''
Author: Eric Aislan Antonelo
email: eric.antonelo@gmail.com
- Part of the Minicourse on Echo State Networks
Last updated on December 2017
'''

import numpy as np
import Oger
import mdp.nodes

from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, subplot, show
import matplotlib.dates as mdates

import pandas as pd
import math
import datetime

## - load daily btc trade data

f = open("data/btc_daily.npz", "rb")
vec = np.load(f)
timestamp, opening, maximum, minimum, close, volume, volume_usd, w_price = 0, 1, 2, 3, 4, 5, 6, 7

# for more data:  https://blockchain.info/charts

## - plot price
plot(vec[:,w_price], '.')
plt.title('BTC price')
show()

## - make training/test data: fill in the inputs and outputs arrays

ahead = 1   ## look ahead days for prediction
N = vec.shape[0] - ahead
N_inputs = 5
inputs = np.zeros((N, N_inputs))
outputs = np.zeros((N, 1))
for i in range(N):
    inputs[i,0] =
    inputs[i,1] =
    inputs[i,2] =
    inputs[i,3] =
    inputs[i,4] = 
    outputs[i,0] = 

close_price = vec[:-1,close]

## - 
import random
random.seed(2000)
np.random.seed(2000)

##- TRAINING

n_reservoir = 400  
ridge_param = 0.01 

input_dim = inputs.shape[1] 
n_train_samples = int(N*0.7)

# Tell the reservoir to save its states for later plotting 
Oger.utils.make_inspectable(Oger.nodes.LeakyReservoirNode)

# construct individual nodes
reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=n_reservoir,  spectral_radius=1.1, input_scaling=.1, leak_rate=1, bias_scaling=.1)

# concatenates the input signal to the reservoir states for training
identity = mdp.nodes.IdentityNode(input_dim=input_dim)
reservoir_and_input = mdp.hinet.SameInputLayer([identity, reservoir])

# creates readout node
readout = Oger.nodes.RidgeRegressionNode(ridge_param=ridge_param)
washout = 20
Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, washout)

# creates the flow 
flow = mdp.Flow([reservoir_and_input, readout], verbose=True)

print "Training..."
flow.train([[], [(inputs[0:n_train_samples,:], outputs[0:n_train_samples,:])] ])
print "Done."

##- 
print "Applying to dataset..."
ytrain = flow(inputs[0:n_train_samples,:])
ytest = flow(inputs[n_train_samples:,:])

testerr = Oger.utils.rmse(ytest[washout:,:], outputs[n_train_samples+washout:,:])
trainerr = Oger.utils.rmse(ytrain[washout:,:], outputs[washout:n_train_samples,:])

print("Train RMSE: {}, Test RMSE: {}".format(trainerr, testerr))

##- evaluation: compute number of agreements in direction between prediction and target
def evaluate(output, target):
    N_test = target.shape[0]
    rplus = np.logical_and(output >= 0, target >= 0)
    rminus = np.logical_and(output < 0, target < 0)
    tplus = np.sum(rplus)
    tminus = np.sum(rminus)
    err = Oger.utils.rmse(output, target)

    print "\nAgree +:", tplus, tplus/float(N_test)
    print "Agree -:", tminus, tminus/float(N_test)
    print "Total Agree:", (tplus + tminus), (tplus + tminus)/float(N_test)
    print "RMSE:", err

# test
print '\ntest'
evaluate(ytest[washout:,0], outputs[washout+n_train_samples:,0])
# train
print '\ntrain'
evaluate(ytrain[washout:,0], outputs[washout:n_train_samples,0])


## -
plot(reservoir.inspect()[0]) # - 1
plt.title('Reservoir states')
plt.show()

## -
plot(inputs)
plt.title('inputs')
show()

plot(outputs)
plt.title('outputs')
show()

## -  plot test
fig, ax1 = plt.subplots()
ax1.plot(outputs[washout+n_train_samples:,:])
ax1.plot(ytest[washout:,:],'r')
# ax1.hlines(0, 0, ytest.shape[0] )
# ax1.plot( 0.2*(ytest[washout:,:] > 0) , '.')
# plt.ylim((-0.4, 0.4))

# ax2 = ax1.twinx()
# ax2.plot(vec[washout+n_train_samples:-2, close], 'k')
plt.title('test outputs')
show()

## plot train
plot(outputs[washout:n_train_samples,:])
plot(ytrain[washout:,:], 'r')
plt.hlines(0, 0, ytrain.shape[0] )
plt.title('train outputs')
show()

## - Test if is better than dumb model
delta_i = 1

targets = outputs[washout+n_train_samples:,:]
closes = close_price[washout+n_train_samples:]
use_model = True
model = ('ESN' if use_model else 'Dumb')

if use_model:
    predictions = ytest[washout-delta_i:,:]
    start_pred = 1
else: # copy reference model
    # - copy predictor: uses previous day percent change as prediction
    predictions = outputs[n_train_samples + washout -2:-1,:]
    start_pred = 2

pred_closes = np.zeros( predictions.shape )
for i,percent in enumerate(predictions):
    if i < start_pred:
        next
    pred_closes[i] = closes[i-delta_i] * (1+percent)

#
print '\n Evaluation on test. Model: ', model
evaluate(predictions[1:], targets)

plot(closes,'k.-')
plot(pred_closes,'r.-')
plt.title('Close price. Model: ' + model )
show()

