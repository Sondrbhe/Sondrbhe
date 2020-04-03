from casadi import *
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as spio
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
#from keras.models import Model
#from keras.layers import Input
#from keras.layers import Dense


#V_in = spio.loadmat('Vin', squeeze_me=True) 
#V_out = spio.loadmat('Vcc', squeeze_me=True) 
#
#Vin_F = V_in['Vin1']
#Vout_F = V_out['Vcc1']


V_in = spio.loadmat('Vin3', squeeze_me=True) 
V_out = spio.loadmat('Vcc3', squeeze_me=True) 

Vin_F = V_in['Vin']
Vout_F = V_out['Vcc']



#plt.figure(1)
#plt.plot(output_vcc['Vcc1'])
#plt.figure(2)
#plt.plot(input_vin['Vin1'])

# Dele inn i trening og test
x_train = np.zeros(shape=(40000,1))
y_train = np.zeros(shape=(40000,1))

x_test = np.zeros(shape=(200,1))
y_test = np.zeros(shape=(200,1))

for i in range(39990):
    x_train[i] = Vin_F[i+10]
    y_train[i] = Vout_F[i+10]
    
for i in range(200):
    x_test[i] = Vin_F[40000+i]
    y_test[i] = Vout_F[40000+i]
    
    
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#mini = min(x_train)
#maxi = max(x_train)

#nomalized = 

maks = max(y_train)
mini = min(y_train)
maks_2 = max(x_train)
min_2 = min(x_train)

for i in range(len(y_train)):
    y_train[i] = (y_train[i] - mini)/(maks - mini) 
    x_train[i] = (x_train[i] - min_2)/(maks_2 - min_2) 

 
for i in range(len(y_test)):
    y_test[i] = (y_train[i] - mini)/(maks - mini) 
    x_test[i] = (x_train[i] - min_2)/(maks_2 - min_2) 
    
    

#NETTERVERKET
model = tf.keras.models.Sequential()
#seqmodel = tf.keras.models.Model()
# model.add(tf.keras.layers.Dense(8, input_dim=1,  activation=tf.nn.tanh))
# model.add(tf.keras.layers.Dense(8, activation=tf.nn.tanh))
# model.add(tf.keras.layers.Dense(8, activation=tf.nn.tanh))
# model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu)) #Output

model.add(tf.keras.layers.Dense(4, input_dim=1,  activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu)) #Output

#
#visible = Input(shape=(2,))
#hidden = Dense(1)(visible)
#
#model = Model(inputs=visible, outputs=hidden)





#TRENING PARAMETERE
model.compile(optimizer ='adam',
              loss='mse', 
              metrics = ['mse'])

#STARTER TRENINGEN
history = model.fit(x_train, y_train, validation_split=0.1, epochs = 55, batch_size=5, verbose=1)


#PREDIKERE 
classes = model.predict(x_test)
#model.save('test2.h5')
#model.save_weights('testWeights.h5')
#test_predictions = model.predict(x_test).flatten()
#test_predictions.save('test2.h5')
#PLPT
plt.plot(classes)
plt.ylabel("TEST")
plt.xlabel("Time")
plt.plot(y_test)
plt.legend(["Predicted", "True"])
plt.show()


#plt.plot(history.history['mean_squared_error'])
# plt.title('mean squared error')
# plt.ylabel('mse')
# plt.xlabel('epoch')
# plt.legend(['val'], loc='upper left')
# plt.show()
