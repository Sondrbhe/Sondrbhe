# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:14:04 2020

@author: Sondr
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
#from control import *
from casadi import *
from Juan_pnmpc import *
import cvxopt
#this is a small crash course on how to run PNMPC control....

print("If the error IDA_LINESEARCH_FAIL, run the program more times. It comes from the model gets negative values")

#first, we need a system, so I created a discrete linear model as a toy problem.
class linear_discrete(SimulationModel):
    
    def __init__(self):
        #MODEL CONSTANTS:
        g = 9.81 #Gravity
        Cc = 2*10**(-5) #Chove valve constant
        A1 = 0.008107 #Cross section below ESP
        A2 = 0.008107 #Cross section above ESP
        D1 = 0.1016 #Pipe diameter below ESP
        D2 = 0.1016 #Pipe diameter above ESP
        h1 = 200 #Height from reservoir to ESP
        h2 = 800 #Height ESP up
        hw = 1000 #Total distance in well
        L1 = 500 #Length reservoir to ESP
        L2 = 1200 #Length from ESP to choce
        V1 = 4.054 #Pipe volume below ESP
        V2 = 9.729 #Pipe volum above ESP
        #ESP DATA 
        B1 = 1.5*10**9 #Bulk modulus below ESP
        B2 = 1.5*10**9 #Bulk modulus above ESP
        M = 1.992*10**8 #Fluid inertia
        rho = 950 #Density of fluid
        pr = 1.26 * 10 ** 7 #Reservoir pressure
        f0 = 60 # ESP reference freq
        #UNKNOWN
        PI = 2.32*10**(-9) #Well producity index
        pm = 20e5 #Manifold pressure
        my = 0.025 # Vicosity of produced fluid
        
        # VCF for ESP flow rate
        CQ = 1 - 2.6266*my + 6.0032*my**2 - 6.8104*my**3 + 2.7944*my**4
        CH = 1 - 0.03*my
        
        #INITIAL VALUES
        self.pbh0 = 70e5
        self.pwh0 = 30e5
        self.q0 = 0.01
        #END CONSTANTS
        
        #INITIALIZE MODEL
        dae = DaeBuilder()
        # Add input expressions
        pbh = dae.add_x('pbh')
        pwh = dae.add_x('pwh')
        q = dae.add_x('q')
        
        qr = dae.add_z('qr')
        qc = dae.add_z('qc')
        
        F1 = dae.add_z('F1')
        F2 = dae.add_z('F2')
        DeltaPp = dae.add_z('DeltaPp')
        
        u = dae.add_p('u')
        f = dae.add_p('f')
        
        # Add output expressions
        pbhdot = (B1/V1)*(qr-q)
        pwhdot = (B2/V2)*(q-qc)
        qdot = (1/M)*(pbh-pwh-rho*g*hw - F1 - F2 + DeltaPp)
        
        dae.add_ode('pbhdot', pbhdot)
        dae.add_ode('pwhdot', pwhdot)
        dae.add_ode('qdot', qdot)
        
        #Algebraic equation
        dae.add_alg('qr', qr-PI*(pr-pbh))
        dae.add_alg('qc', qc-Cc*(sqrt(fmax(10**-5, pwh-pm))*u))
        
        dae.add_alg('F1', F1 - 0.158*((rho*L1*q**2)/(D1*A1**2))*(my/(rho*D1*q))**(1/4))
        dae.add_alg('F2', F2 - 0.158*((rho*L2*q**2)/(D2*A2**2))*(my/(rho*D2*q))**(1/4))
        
        # Algebraic ESP equations:
        dae.add_alg('DeltaPp', DeltaPp - rho*g* (CH*(9.5970e2 + 7.4959e3*((q/CQ)*(f0/f)) - 1.2454e6*((q/CQ)*(f0/f))**2)*(f/f0)**2))
        #CREATE DAE
        dae = {'x': vertcat(*dae.x), 'p': vertcat(*dae.p),'z': vertcat(*dae.z), 'ode': vertcat(*dae.ode), 'alg': vertcat(*dae.alg)}
        ts = np.linspace(0,0.1,3)
        opts = {'grid':ts}
        self.I = integrator('I', 'idas', dae, opts)
        

        
    	#START MED INITAL, så gå over til first result
    def model_output(self,u):
        I = self.I
#        print(u[1])
        z = u[0]
        f = u[1]
      #FIRST STEP WITH INPUT z =1 , f = 40
        x0 = vertcat(self.pbh0,self.pwh0,self.q0) #DIFFERENTIAL
#        print(x0)
        p0 = vertcat(z,f) #INPUTS
        res = self.I(x0=x0,p=p0)
        solution = res['xf'].full().T
        
        self.pbh0 = solution[len(solution)-1,0] #Initalizing model with correct states with new input
        self.pwh0 = solution[len(solution)-1,1]
        self.q0 = fmax(1e-5,solution[len(solution)-1,2])
        y = solution[len(solution)-1,:]
        return y
    
    

ESP = linear_discrete()
#------------------------------------------



#we first need to gather data, so we use the built-in RFRAS (Random Frequency Random Amplitude Signal)


#these lines of code create the identificaton signal:
simtime = 2000
u_min = np.array([0.6])
u_max = np.array([1])
some_input_u = RFRAS(u_min,u_max,simtime,30)

f_min = np.array([50])
f_max = np.array([60])
some_input_f = RFRAS(f_min,f_max,simtime,30)
#print(some_input)
#THe data MUST be 2dimensional, and data must organized in row form, so in this case:

some_input_u = np.atleast_2d(some_input_u).T
some_input_f = np.atleast_2d(some_input_f).T

some_input = np.zeros([len(some_input_u),2])

for i in range (len(some_input_u)):
    some_input[i,0] = some_input_u[i]
    some_input[i,1] = some_input_f[i]

#some_input = [*some_input_u , *some_input_f]

#the resulting dimension should always be (number of data x number of features)

#now we need to run the system with that signal to obtain the system response, which is recorded in y_data
y_data = np.empty([simtime,3])
y_data[0] = (70e5,30e5,0.01)
#the simulation and plotting for data gathering is as follows:
for i in range(simtime):
    y = ESP.model_output(some_input[i])
    y_data[i] = y
#print(y_data)


#PLOTS OF THE DATA INCLUDED THE INPUTS
plt.figure(1)
plt.subplot(511)
plt.plot(y_data[:,0]/10**5,color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')


plt.subplot(512)
plt.plot(y_data[:,1]/10**5,color='blue',label='$p_{wh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(513)
plt.plot(y_data[:,2]*3600,color='blue',label='$q$')
plt.grid()
plt.legend()
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')
plt.xlabel('Time [min]')

plt.subplot(514)
plt.plot(some_input[:,0],color='red',label='$valve$')
plt.grid()
plt.legend()
plt.xlabel('Time [min]')

plt.subplot(515)
plt.plot(some_input[:,1],color='red',label='$frekvens$')
plt.grid()
plt.legend()
plt.xlabel('Time [min]')

##plt.plot(y_data)
##------------------------------------------------------------


#next step is to normalize the data using feature scaling, so we need to use
#a minimum value and a maximum value, even better if it ties to the constraints of your problem.
#
##The scaling is done as follows, using the built_in function from esn_pnmpc_library:
u_data_normal = feature_scaling(some_input,u_max,u_min)
y_max = np.max(y_data,axis=0)
y_min = np.min(y_data,axis=0)
y_data_normal = feature_scaling(y_data,y_max,y_min)


u_data_normal = np.zeros([len(some_input_u),2])
y_data_normal = np.zeros([len(some_input_u),2])

u_data_normal[:,0] = feature_scaling(some_input[:,0],u_max,u_min)
u_data_normal[:,1] = feature_scaling(some_input[:,1],f_max,f_min)


y_max = np.max(y_data,axis=0)
y_min = np.min(y_data,axis=0)
y_data_normal = feature_scaling(y_data,y_max,y_min)


##---------------------------------------------------------------------------------------------------
#
##Now, ater obtaining a set of normalized inputs and outputs, we shall instantiate and train a black-box model:
#
##in this case, an ESN is used, with the parameters arbitrariely selected.
##Obviously, the ESN is a bit overkill for the problem of training a linear third order system...
##
import ESN_JUAN as RNN
#toy_esn = RNN.EchoStateNetwork(neu = 500,
#                           n_in = 2,
#                           n_out = 3,
#                 gama=0.8,
#                 ro=0.99,
#                 psi=0.0,
#                 in_scale=0.1,
#                 bias_scale=0.1,
#                 initial_filename="initial",
#                 load_initial = True,
#                 save_initial = False,
#                 noise_amplitude = 5e-3)
toy_esn = RNN.EchoStateNetwork(neu = 100,n_in = 2,n_out = 3,gama = 0.14,ro = 0.99,psi = 0.0,in_scale = 0.1,bias_scale = 0.1, initial_filename = "ESPmodel", load_initial = False, save_initial = False,output_feedback = False)
#-----------------------------------------------------------------------------------------------


#now, we need to train it, by first feeding the ESN with the normalized data.
#the warmupdrop means that the first few [warmupdrop] examples will the dropped.
#In this case, the first few examples are outliers and have a high probability of corrupting the data.

#the function add data serves to gather data of one simulation run to the class "database".
#Multiples simulation could also be absorbed by multiple runs of the command.

#now, we train. The training function also returns the best regularization parameter while insternally changing the weights.

regularization = 1e-8	 
drop = 200

toy_esn.offline_training(u_data_normal,y_data_normal,regularization,drop)
 

 	 
#adter training the ESN, it is recommended you generate another input signal to test the ESN:

simtime = 500

#
#CREATING TEST INPUT WITH RANDOM IN
test_input_u = RFRAS(u_min,u_max,simtime,30)
test_input_f = RFRAS(f_min,f_max,simtime,30)

test_input = np.zeros([len(test_input_u),2])

for i in range (len(test_input_u)):
    test_input[i,0] = test_input_u[i]
    test_input[i,1] = test_input_f[i]

#CREATING THE PREDICTION WITH ESN 
y_pred_test = np.empty([simtime,3])
y_test = np.empty_like(y_pred_test)

for k in range(simtime):
    y_test[k] = ESP.model_output(test_input[k])


test_input[:,0] = feature_scaling(test_input[:,0],u_max,u_min)
test_input[:,1] = feature_scaling(test_input[:,1],f_max,f_min)


for k in range(simtime):
    y_pred_test[k] = toy_esn.update(test_input[k]).flatten()
    y_pred_test[k] = feature_descaling(y_pred_test[k],y_max,y_min)


#PLOTS WITH REAL AND PREDICTED VALUES
plt.figure(2)
plt.subplot(511)
plt.plot(y_test[:,0]/10**5,color='blue', label='$p_{bh}$')
plt.plot(y_pred_test[:,0]/10**5,color='orange', label='$pred$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')

plt.subplot(512)
plt.plot(y_test[:,1]/10**5,color='blue', label='$p_{wh}$')
plt.plot(y_pred_test[:,1]/10**5,color='orange', label='$pred$')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(513)
plt.plot(y_test[:,2]*3600,color='blue', label='$q$')
plt.plot(y_pred_test[:,2]*3600,color='orange', label='$pred$')
plt.grid()
plt.legend()
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')

plt.subplot(514)
plt.plot(test_input[:,0],color='red',label='$valve$')
plt.grid()
plt.legend()

plt.subplot(515)
plt.plot(test_input[:,1],color='red',label='$frekvens$')
plt.grid()
plt.legend()
plt.xlabel('Time [min]')
#from the plot it seems the esn can easily copy the system behavior, therefore we are ready to instantiate the predictor:

a = 0.5 #forcing the pole of the filter.

toy_predictor = PNMPCPredictor(control_horizon = 5,
                               prediction_horizon = 30,
                               bb_model = toy_esn,
                               filter_constants = a**2, 
                               K = (a**2 - 2.0*a +1)/(1 - a**2))

#now that we just created the predictor, we need to create the control instance....
initial_u = [[0.7],[40]]
print(initial_u)
control = PNMPCController(n_cv = 2,
                 u_max = u_max,
                 u_min = u_min,
                 y_max = y_max,
                 y_min = y_min, 
                 initial_u = initial_u,
                 Q = np.eye(3),
                 R = np.eye(2),
                 du_max_c = 0.6,
                 output_constraints_max = 1*y_max[1:],
                 output_constraints_min = 1*y_min[1:],
                 predictor = toy_predictor,
                 simulation_model = ESP)


#running the controller is simple, you just have to define a proper reference signal for it, of dimension n_cv
pnmpc_time = 500
#y_ref = RFRAS(y_min[0],y_max[0],pnmpc_time,20)
#reference should be organized just like the data matrices, this is needed because there is only one controlled variable:
#y_ref = np.atleast_2d(y_ref).T

y_ref = np.zeros([pnmpc_time,2])

for i in range (pnmpc_time):
    y_ref[i,0] = 50
    y_ref[i,1] = 30


warm_up_time = 200


output_dict = control.run_controller(y_ref) #THE ERROR COMES HERE
#



plt.figure(1)
plt.subplot(411)
plt.plot(y_ref,color='blue', label='$reference$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')

plt.subplot(412)
plt.plot(output_dict['y_plot'][:,0],color='blue',label='$p_{wh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(413)
plt.plot(output_dict['y_plot'][:,1],color='blue',label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(414)
plt.plot(output_dict['y_plot'][:,2],color='blue',label='$q$')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

sub[0].plot(y_ref)
sub[0].plot(output_dict['y_plot'][:,0])


plt.plot(output_dict['correction_plot'])

#sub[1].plot(output_dict['y_plot'][:,1])
#sub[2].plot(output_dict['y_plot'][:,2])
ax = plt.gca()
ax.set_xlabel("time")
sub[0].set_ylabel("CV 1")
sub[1].set_ylabel("constraint 1")
sub[2].set_ylabel("constraint 2")

plt.show()

print("IAE:",output_dict['IAE'])