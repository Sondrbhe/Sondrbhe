# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:26:54 2020

@author: iver_
"""


import matplotlib.pyplot as plt
from casadi import *
import numpy as np


def shift(T, t0, x0, u, f):
    st = x0
    con = u[0,:].T
    f_value = f(st,con)
    st = st + (T*f_value)
    x0 = st.full()
    
    t0 = t0 + T
    a1 = u[1:len(u)]
    a2 = u[len(u)-1]
    u0 = np.vstack((a1, a2))
    return t0, x0, u0


simtime = 50
T = 0.2 # Samlingtime
N = 5 # Prediction horizon
length = int(simtime/T)

u_max = 1
u_min = 0

x1 = SX.sym('x1')
x2 = SX.sym('x2')
states = vertcat(x1, x2)
n_states = states.size1()

u = SX.sym('u')
controls = u
n_controls = controls.size1()
rhs = vertcat((1-x2**2)*x1 - x2 + u, x1)  #Right hand side of eq


f = Function("f", [states,controls],[rhs]) # Nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N) # Declaration variables (control). Prediction.
P = SX.sym('P',n_states + n_states) # parameters (which include the initial and the reference state of the watertank)

X = SX.sym('X',n_states,(N+1)) # A matrix that represents the states over the optimization problem. Prediction

# Compute solution symbolically
X[:,0] = P[0:2] # Init state

# Filling up stateprediction matrix
for k in range(0,N):
    st = X[:,k]
    con = U[:,k]
    f_value = f(st,con) # This function gives us the right hand side.
    st_next = st + (T*f_value)
    X[:,k+1] = st_next


ff = Function('ff',[U,P], [X]) # Gives prediction of X.jj

obj = 0 # Objective function
g = [] # Constraints vector

## Tuning parameters
Q = np.zeros((2,2))
Q[0,0] = 1
Q[1,1] = 1

R = 0.01

# test1 = 
# test1 = (st-P[2:4]) @ Q , MÅ SKRIVE @ FOR Å MULTIPLISERE MATRISER FFFFS
# test2 = test1.T
# test = test2*Q#*(st-P[2:4])
#Q = DM.eye(2)
#tes1 = st * Q 
# computing objective
for k in range(0,N):
    st = X[:,k]
    con = U[:,k]
    obj = obj + (st-P[2:4]).T @ Q @ (st-P[2:4]) + con.T @ R @ con
    #obj =  obj + (st-P[2:4])*Q*(st-P[2:4]) #+ con.T*R*con # Calculate obj. Sum, derfor i forloop er referanse

# Constraints kjem her.
# g = ...
    
# Make the decision variables one column vector
# OPT_variables = reshape(U,2*N,1); Må bli med om vi har meir enn en kontrollvariabel
nlp_prob = {'f':obj,'x':U, 'g': g, 'p': P}


opts = {'ipopt':{'max_iter':100, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6 }, 'print_time':0}


#opts = {'ipopt.max_iter':simtime, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
# opts.ipopt.max_iter = simtime
# opts.ipopt.print_level = 0
# opts.print_time = 0
# opts.ipopt.acceptable_tol = 1e-8
# opts.ipopt.acceptable_obj_change_tol = 1e-6

solver = nlpsol('solver','ipopt', nlp_prob, opts)


# Input constraints (Commented out, think I can fix this later on)
#args.lbx(1:1:N,1) = u_min;
#args.ubx(1:1:N,1) = u_max;

#-------------------------------------------------------------------------
## THE SIMULATION LOOP STARTS FROM HERE
t0 = 0
#x0 = [0,1]
#xs = [0,0]
x0 = vertcat(0,1)
xs = vertcat(0,0)
xx = np.zeros([2,length+1])
xx[0,0] = 0
xx[1,0] = 1 # xx contains the history of states.
t = np.zeros([length])
t[0] = 0

u0 = np.zeros([N,1])
u0[:,0] = 0
#u0 = zeros(N,1) # One control input.
sim_tim = simtime # Maximum simulation time.xxl

#  Creating a dict where all info is stored
args = {'lbx':np.full((N,1), u_min), 'ubx':np.full((N,1), u_max)}

# Start MPC
mpciter = 0 # Counter for the loop
#xxl = np.zeros([N+1,1,length])
xxl = np.zeros([N+1,2,length])
u_cl = np.zeros([length,1])
#main_loop = tic; # Find something later to measure the time spent.

while(mpciter < sim_tim / T):
    # args.p = [x0, xs] # set the values of the parameters vector
    # %args.x0 = reshape(u0.T,2*N,1) # initial value of the optimization variables (x0)
    # args.x0 = u0';
    
    # p skal ligge på rekke og rad, NEDOVER! :)
    # Dette blir på en måte en ny funksjon, sjekk ut docs
    sol = solver(x0=u0.T, p=vertcat(x0,xs), lbx=args['lbx'], ubx=args['ubx'])
    # sol = solver('x0', args.x0, 'p', args.p, 'lbx', args.lbx, 'ubx', args.ubx) #'lbx', args.lbx, 'ubx', args.ubx,...
    #     #'lbg', args.lbg, 'ubg', args.ubg,...
   # u = np.reshape(sol['x'].full().T,1,N).T#  optimal values for minimize control
    u = np.reshape(sol['x'].full().T, (1,N)).T
    #ff_value = ff(u.T,[x0,xs]) #compute OPTIMAL solution TRAJECTORY
    ff_value = ff(u.T,vertcat(x0,xs))
    xxl[:,0:2,mpciter] = ff_value.full().T
    u_cl[mpciter] = u[1,:]
    
    t[mpciter] = t0
    [t0, x0, u0] = shift(T, t0, x0, u, f)
    #xx[:,mpciter+1] = x0
    xx[0,mpciter+1] = x0[0]
    xx[1,mpciter+1] = x0[1]
    mpciter = mpciter +1
    
    
plt.figure()
plt.subplot(311)
plt.step(t,u_cl[:,0],color='red', label='$u$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')

plt.subplot(312)
plt.plot(xx[0,:],color='blue', label='$\dot h$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')

plt.subplot(313)
plt.plot(xx[1,:],color='blue', label='$\dot h$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')
    

