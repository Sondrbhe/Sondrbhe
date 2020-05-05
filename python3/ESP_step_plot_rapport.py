

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:50:33 2020
@author: Sondr
"""
import matplotlib.pyplot as plt
from casadi import *
import numpy as np
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
dae.set_start('q', 0.013)
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
valve = [0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]
dae = {'x': vertcat(*dae.x), 'p': vertcat(*dae.p),'z': vertcat(*dae.z), 'ode': vertcat(*dae.ode), 'alg': vertcat(*dae.alg)}
#ts = np.linspace(0,10,300) # Her ligg feilen!! :)
ts = np.linspace(0,1,10+1) 
opts = {'grid':ts}
#opts = {"tf":1} # interval length
I = integrator('I', 'idas', dae, opts)
x0 = vertcat(80e5,30e5,0.01)
p0 = vertcat(valve[0],53)
res = I(x0=x0,p=p0)
final_1 = res['xf'].full().T
temp = final_1[len(final_1)-1]
x0 = vertcat(temp)
p0 = vertcat(valve[1],53)
res = I(x0=x0,p=p0)
final_2 = res['xf'].full().T
solution = np.append(final_1,final_2,axis=0)
#valve = np.linspace(0.3,1,10)
for i in range(2,10):
    temp = final_2[len(final_1)-1]
    x0 = vertcat(temp)
    p0 = vertcat(valve[i],53)
    res = I(x0=x0,p=p0)
    final_2 = res['xf'].full().T
    temp = final_2[len(final_1)-1]
    solution = np.append(solution,final_2,axis=0)
time = np.linspace(0,10,2990)
time_1 = np.linspace(0,10,10)
plt.figure(1)
plt.grid()
plt.subplot(411)
plt.grid()
x_1 = plt.plot(solution[:,0]/10**5,label='$p_{bh}$')
plt.legend()
plt.xlabel('Time [min]')
plt.ylabel('$p_{bh} \ [bar]$')
plt.subplot(412)
x_2 = plt.plot(solution[:,1]/10**5,label='$p_{wh}$')
plt.legend()
plt.grid()
plt.xlabel('Time [min]')
plt.ylabel('$p_{wh} \ [bar]$')
plt.subplot(413)
x_3 = plt.plot(solution[:,2]*3600,label='q')
plt.legend()
plt.grid()
plt.xlabel('Time [min]')
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')
plt.subplot(414)
x_4 = plt.step(time_1,valve,color='red',label='Valve opening')
plt.grid()
plt.legend()
plt.ylabel('$z \ \% $')
plt.xlabel('Time [min]')
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