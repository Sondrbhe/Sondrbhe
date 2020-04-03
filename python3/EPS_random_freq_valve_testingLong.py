# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:50:33 2020

@author: Sondr
"""

import matplotlib.pyplot as plt
from casadi import *
import numpy as np
import scipy.io

#KNOWN CONTANTS
g = 9.81
Cc = 2*10**(-5)
A1 = 0.008107
A2 = 0.008107
D1 = 0.1016
D2 = 0.1016
h1 = 200
h2 = 800
hw = 1000
L1 = 500
L2 = 1200
V1 = 4.054
V2 = 9.729
#ESP DATA
B1 = 1.5*10**9
B2 = 1.5*10**9
M = 1.992*10**8
rho = 950
pr = 1.26 * 10 ** 7
f0 = 60 # Hz
#f = [57, 57, 54, 64, 64, 49, 49, 51, 53, 54]

#f = 53 # Hz

#UNKNOWN
PI = 2.32*10**(-9)

#KANSKJE
pm = 20e5

my = 0.025 # Pa*s



#DELTA
#u = 1
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
# VCF for ESP flow rate
CQ = 1 - 2.6266*my + 6.0032*my**2 - 6.8104*my**3 + 2.7944*my**4
CH = 1 - 0.03*my



dae.add_alg('DeltaPp', DeltaPp - rho*g* (CH*(9.5970e2 + 7.4959e3*((q/CQ)*(f0/f)) - 1.2454e6*((q/CQ)*(f0/f))**2)*(f/f0)**2))


# Specify initial conditions
dae.set_start('pbh', 70e5)
dae.set_start('pwh', 30e5)
dae.set_start('q', 0.010)
dae.set_start('qr', 0)
dae.set_start('qc', 0)
dae.set_start('F1', 0)
dae.set_start('F2', 0)
#dae.set_start('ppin', 0)
#dae.set_start('ppout', 0)
#dae.set_start('alg3', 1)
# Add meta information
#dae.set_unit('h','m')
#dae.set_unit('v','m/s')
#dae.set_unit('m','kg')


#################

nstep = 10000

# random signal generation
a = np.random.randint(40, 100, nstep)  # range for amplitude
b = np.random.randint(1, 5, nstep) # range for freuency

b[0] = 0

for i in range(1,np.size(b)):
    b[i] = b[i-1]+b[i]

# Random Signal
i=0
random_signal_valve = np.zeros(nstep)
#print(random_signal)
while b[i]<np.size(random_signal_valve):
    k = b[i]
    random_signal_valve[k:] = a[i]
    i=i+1


random_signal_valve = random_signal_valve/100
print(random_signal_valve/100)

###################
# random signal generation
a = np.random.randint(35, 65, nstep)  # range for amplitude
b = np.random.randint(1, 5, nstep) # range for freuency


b[0] = 0

for i in range(1,np.size(b)):
    b[i] = b[i-1]+b[i]

# Random Signal
i=0
random_signal_freq = np.zeros(nstep)
while b[i]<np.size(random_signal_freq):
    k = b[i]
    random_signal_freq[k:] = a[i]
    i=i+1
# #############################
# for i in range(nstep):
#     random_signal_freq[i] = 65
#     random_signal_valve[i] = 0.2


i = 0
j = 0
k = 0
x = np.zeros((100000,2))
y = np.zeros((100000,2))

for i in range(0,10000):   
    for j in range(0,6):
        x[k,0] = random_signal_valve[i]
        y[k,1] = random_signal_freq[i]
        k += 1



print(x)




# ##############################
#valve = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
#freq = [40,40,40,40,40,40,40,40,40,40]
valve = random_signal_valve
freq = random_signal_freq

dae = {'x': vertcat(*dae.x), 'p': vertcat(*dae.p),'z': vertcat(*dae.z), 'ode': vertcat(*dae.ode), 'alg': vertcat(*dae.alg)}
ts = np.linspace(0,nstep,7)
opts = {'grid':ts}
#opts = {"tf":1} # interval length
I = integrator('I', 'idas', dae, opts)


x0 = vertcat(70e5,30e5,0.025)
p0 = vertcat(valve[0],freq[0])
res = I(x0=x0,p=p0)
final_1 = res['xf'].full().T
temp = final_1[len(final_1)-1]

x0 = vertcat(temp)
p0 = vertcat(valve[1],freq[0])
res = I(x0=x0,p=p0)
final_2 = res['xf'].full().T


solution = np.append(final_1,final_2,axis=0)
print(solution)


#valve = np.linspace(0.3,1,10)


for i in range(2,nstep):
    temp = final_2[len(final_1)-1]
    x0 = vertcat(temp)
    p0 = vertcat(valve[i],freq[i])
    res = I(x0=x0,p=p0)
    final_2 = res['xf'].full().T
    temp = final_2[len(final_1)-1]
    solution = np.append(solution,final_2,axis=0)

time = np.linspace(0,nstep,len(solution))
time_1 = np.linspace(0,nstep-1,nstep)
############
    
# plt.figure(1)
# plt.subplot(511)
# plt.plot(time,solution[:,0]/10**5,color='blue', label='$p_{bh}$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{bh} \ [bar]$')


# plt.subplot(512)
# plt.plot(time,solution[:,1]/10**5,color='blue',label='$p_{wh}$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{wh} \ [bar]$')

# plt.subplot(513)
# plt.plot(time,solution[:,2]*3600,color='blue',label='$q$')
# plt.grid()
# plt.legend()
# plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')

# plt.subplot(514)
# plt.step(time_1,valve,color='red',label='$valve$')
# plt.grid()
# plt.legend()
# plt.ylabel(r'$valve$')

# plt.subplot(515)
# plt.step(time_1,freq,color='red',label='$freq$')
# plt.grid()
# plt.legend()
# plt.ylabel(r'$freq$')
# plt.xlabel('Time [min]')
# ################3


#input1 = np.zeros((nstep*6,2))
for i in range(0,nstep):
    for j in range(0,6):
        input1 = np.append(valve[i], freq[i], axis=0)
    


# Lagre til .mat fil
#scipy.io.savemat('ESPdatasetv1.mat', {'output': solution})

#plt.figure(1)
#plt.grid()
#plt.plot(time,solution[:,0]/10**5,label='$p_{bh}$')
##plt.plot(solution[:,0]/10**5,label='pbh')
#plt.axvline(x=5, ymin=0, ymax=80,color='red',linestyle=':',label='Valve opening $\Longrightarrow$ 100%')
#plt.legend()
#plt.xlabel('Time [min]')
#plt.ylabel('$p_{bh} \ [bar]$')
#
#
#
#plt.figure(2)
#plt.plot(time,solution[:,1]/10**5,label='$p_{wh}$')
#plt.axvline(x=5, ymin=0, ymax=80,color='red',linestyle=':',label='Valve opening $\Longrightarrow$ 100%')
#plt.legend()
#plt.grid()
#plt.xlabel('Time [min]')
#plt.ylabel('$p_{wh} \ [bar]$')
#
#
#
#plt.figure(3)
#plt.plot(time,solution[:,2]*3600,label='q')
#plt.axvline(x=5, ymin=0, ymax=80,color='red',linestyle=':',label='Valve opening $\Longrightarrow$ 100%')
#plt.legend()
#plt.grid()
#plt.xlabel('Time [min]')
#plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')


#



#plt.plot(test2[:,0],label='z')
#plt.plot(test2[:,1],label='z1')
#plt.plot(test2[:,2],label='z2')
#plt.plot(test2[:,3],label='z3')
#plt.plot(ts,xf[1,:].T,label='x1')
#plt.plot(ts[1:],zf.T,label='z')


#
#my = 0.3
#
#
## Viscosity Correction Factors (VCF)
## VCF for head
#CH = 1 - 0.03*my
#
## VCF for brake horsepower of the ESP
#CP = 1 + 3.9042*my - 9.9306*my**2 + 11.091*my**3 - 4.4376*my**4
#
## VCF for ESP flow rate
#CQ = 1 - 2.6266*my + 6.0032*my**2 - 6.8104*my**3 + 2.7944*my**4
#
## ESP head characteristics
#H0 = 9.5970e2 + 7.4959e3*q - 1.2454e6*q**2
#
## ESP BHP characteristics
#P0 = 9.4355e4 + 4.3346e6*q - 1.9092e7*q**2 - 2.3599e9*q**3


#array = [57, 57, 54, 64, 64, 49, 49, 51, 53, 54]
#plt.plot(array)




# All controls
#U = MX.sym("U", 20)
#
## Construct graph of integrator calls
#X  = [0,1]
#J = 0
#for k in range(20):
#  Ik = I(x0=X, p=U[k])
#  X = Ik['xf']
#  J += Ik['qf']   # Sum up quadratures
#
#
#











