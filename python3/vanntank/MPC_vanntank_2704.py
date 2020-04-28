# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:22:01 2020

@author: iver_
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:02:54 2020

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



# Paramteres
q_in = 5
Area = 20
C = 0.15
g = 9.81
h0 = 10
rho = 1000
h_ref = 5

simtime = 20
T = 0.2 # Samlingtime
N = 3 # Prediction horizon
length = int(simtime/T)

u_max = 1
u_min = 0

x = SX.sym('x')
states = x
n_states = states.size1()

u = SX.sym('u')
controls = u
n_controls = states.size2()
rhs = (1/Area)*(q_in - C*u*np.sqrt(rho*g*x)) # Right hand side of eq


f = Function("f", [states,controls],[rhs]) # Nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N) # Declaration variables (control). Prediction.
P = SX.sym('P',n_states + n_states) # parameters (which include the initial and the reference state of the watertank)

X = SX.sym('X',n_states,(N+1)) # A matrix that represents the states over the optimization problem. Prediction

### 52:58
# Compute solution symbolically
# This one causes error when set to X[:,0] = P[0], but I think it is the right one.
X[:,0] = P[0]
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
Q = 1 # Weighing matrices (states). (He kun en state).
R = 1 # Weighing matries (control).

# computing objective
for k in range(0,N):
    st = X[:,k];
    con = U[:,k];
    # The calculation below are a bit sketchy with P[1], it worked in matlab, but not sure how it works.
    obj = obj + (st-P[1]).T*Q*(st-P[1]) + (con.T-0.150)*R*(con-0.150) # Calculate obj. Sum, derfor i forloop er referanse


# Constraints kjem her.
# g = ...
    
# Make the decision variables one column vector
# OPT_variables = reshape(U,2*N,1); Må bli med om vi har meir enn en kontrollvariabel
nlp_prob = {'f':obj,'x':U, 'g': g, 'p': P}


opts = {'ipopt':{'max_iter':10, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6 }, 'print_time':0}


#opts = {'ipopt.max_iter':simtime, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
# opts.ipopt.max_iter = simtime
# opts.ipopt.print_level = 0
# opts.print_time = 0
# opts.ipopt.acceptable_tol = 1e-8
# opts.ipopt.acceptable_obj_change_tol = 1e-6

solver = nlpsol("solver","ipopt",nlp_prob,opts)


# Input constraints (Commented out, think I can fix this later on)
#args.lbx(1:1:N,1) = u_min;
#args.ubx(1:1:N,1) = u_max;

#-------------------------------------------------------------------------
## THE SIMULATION LOOP STARTS FROM HERE
t0 = 0
x0 = h0
xs = h_ref
xx = np.zeros([1,length+1])
xx[:,0] = x0 # xx contains the history of states.
t = np.zeros([length])
t[0] = 0

u0 = np.zeros([N,1])
u0[:,0] = 0.150
#u0 = zeros(N,1) # One control input.
sim_tim = simtime # Maximum simulation time.xxl

#  Creating a dict where all info is stored
args = {'lbx':np.full((N,1), u_min), 'ubx':np.full((N,1), u_max)}

# Start MPC
mpciter = 0 # Counter for the loop
#xxl = np.zeros([N+1,1,length])
xxl = np.zeros([N+1,length])
u_cl = np.zeros([length,1])
#main_loop = tic; # Find something later to measure the time spent.

while(mpciter < sim_tim / T):
    # args.p = [x0, xs] # set the values of the parameters vector
    # %args.x0 = reshape(u0.T,2*N,1) # initial value of the optimization variables (x0)
    # args.x0 = u0';
    sol = solver(x0=u0.T, p=[x0,xs], lbx=args['lbx'], ubx=args['ubx'])
    # sol = solver('x0', args.x0, 'p', args.p, 'lbx', args.lbx, 'ubx', args.ubx) #'lbx', args.lbx, 'ubx', args.ubx,...
    #     #'lbg', args.lbg, 'ubg', args.ubg,...
   # u = np.reshape(sol['x'].full().T,1,N).T#  optimal values for minimize control
    u = np.reshape(sol['x'].full().T, (1,N)).T
    ff_value = ff(u.T,[x0,xs]) #compute OPTIMAL solution TRAJECTORY
    xxl[:,mpciter] = ff_value.full()
    u_cl[mpciter] = u[1,:]
    
    t[mpciter] = t0
    [t0, x0, u0] = shift(T, t0, x0, u, f)
    xx[:,mpciter+1] = x0
    mpciter = mpciter +1
    
    
plt.figure()
plt.subplot(211)
plt.step(t,u_cl[:,0],color='red', label='$u$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')

plt.subplot(212)
plt.plot(xx[0,:],color='blue', label='$\dot h$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')
    

