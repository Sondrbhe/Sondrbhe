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


dae = {'x': vertcat(*dae.x), 'p': vertcat(*dae.p),'z': vertcat(*dae.z), 'ode': vertcat(*dae.ode), 'alg': vertcat(*dae.alg)}
ts = np.linspace(0,10,400)
opts = {'grid':ts}
#opts = {"tf":1} # interval length
I = integrator('I', 'idas', dae, opts)

x0 = vertcat(70e5,30e5,0.01)
p0 = vertcat(1,53)
res = I(x0=x0,p=p0)



t = res['xf']


#
solution = res['xf'].full().T




plt.figure(1)
plt.subplot(311)
plt.plot(ts[1:],solution[:,0]/10**5,color='blue', label='$p_{bh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')


plt.subplot(312)
plt.plot(ts[1:],solution[:,1]/10**5,color='blue',label='$p_{wh}$')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(313)
plt.plot(ts[1:],solution[:,2]*3600,color='blue',label='$q$')
plt.grid()
plt.legend()
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')
plt.xlabel('Time [min]')


###########################







