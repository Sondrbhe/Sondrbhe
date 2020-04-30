# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:21:51 2020

@author: iver_
"""


from casadi import *
import matplotlib.pyplot as plt
#DELTA
#u = 1
dae = DaeBuilder()
# Add input expressions
x1 = dae.add_x('x1')
x2 = dae.add_x('x2')
u = dae.add_p('u')
# Add output expressions
x1dot = (1-x2**2)*x1 - x2 + u
x2dot = x1
dae.add_ode('x1dot', x1dot)
dae.add_ode('x2dot', x2dot)
dae = {'x': vertcat(*dae.x), 'p': vertcat(*dae.p), 'ode': vertcat(*dae.ode)}
ts = np.linspace(0,10,500)
opts = {'grid':ts}
#opts = {"tf":1} # interval length
F = integrator('F', 'idas', dae, opts)
x0 = vertcat(0,1)
p0 = vertcat(0)
res = F(x0=x0,p=p0)
final_1 = res['xf'].full().T
#print(final_1[:,0])
#
#
plt.plot(final_1[:,0].T,label='x0')
plt.plot(final_1[:,1].T,label='x1')