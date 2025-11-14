"""
Linear-quadratic regulators (LQRs) were proposed in 
[1] for full-state feedback control and in [2] for 
output-feedback control.
This script is part of the source code of [3] 
and implements the first-order nonlinear
ordinary differential equation of the mean 
tropospheric temperature proposed in [4]. The value
of the average incoming solar energy per unit area,
namely, S, is taken from [5].
       
REFERENCES:
[1] Nersesov, S.G., Haddad, W.M. and Chellaboina, V.,
2004. Optimal fixed‐structure control for linear 
non‐negative dynamical systems. 
International Journal of Robust and Nonlinear 
Control: IFAC‐Affiliated Journal, 14(5), pp.487-511.
[2] Ilka, A. and Murgovski, N., 2022. Novel results 
on output-feedback LQR design. IEEE Transactions on 
Automatic Control, 68(9), pp.5187-5200.
[3] Zocco, F., Haddad, W.M. and Malvezzi, M., 2025. 
CarboNet: A finite-time combustion-tolerant 
compartmental network for tropospheric carbon 
control. arXiv preprint arXiv:2508.16774.
[4] Elsherif, S.M. and Taha, A.F., 2025. Climate 
science and control engineering: Insights, parallels, 
and connections. arXiv preprint arXiv:2504.21153.
[5] NASA website on the value of S: https://earth.gsfc.nasa.gov/climate/projects/solar-irradiance/science      
"""
   
from IPython import get_ipython
get_ipython().magic('reset -sf') # Clear all
###

import numpy as np
import math
from scipy import integrate
import matplotlib.pyplot as plt

# Set the feedback gain to use:
chosen_gain = 'K_of' #set as 'K_fs' for full-state feedback 
#gain or 'K_of' for output-feedback gain 
##############

# Definition of closed-loop matrix based on the gain K:
if chosen_gain == 'K_fs':  
    aHat_11 = -1.13984418  
    aHat_12 = -0.16695025
    aHat_13 = -0.42842023
    aHat_14 = -0.5481119
    aHat_21 = 0.0
    aHat_22 = -0.7
    aHat_23 = 0.0
    aHat_24 = 0.0
    aHat_31 = 0.0 
    aHat_32 = 0.0
    aHat_33 = -0.4
    aHat_34 = 0.0
    aHat_41 = 0.2
    aHat_42 = 0.5
    aHat_43 = 0.5
    aHat_44 = -0.1
    
elif chosen_gain == 'K_of':  
    aHat_11 = -1.037174938693166 
    aHat_12 = 0.500000000000000
    aHat_13 = 0.500000000000000
    aHat_14 = 0.100000000000000
    aHat_21 = 0.0
    aHat_22 = -0.700000000000000
    aHat_23 = 0.0
    aHat_24 = 0.0
    aHat_31 = 0.0 
    aHat_32 = 0.0
    aHat_33 = -0.400000000000000
    aHat_34 = 0.0
    aHat_41 = 0.200000000000000
    aHat_42 = 0.500000000000000
    aHat_43 = 0.500000000000000
    aHat_44 = -0.100000000000000
else:
    print('Gain K is not valid')
    

# Parameters of temperature model:
C = 8*10**8 #taken from [4]
S = (1361.0)*(86400)*(1/4) #taken from [5]; 86400 is the conversion from seconds to days
alpha = 0.3 #taken from [4]
epsilon_n = 0.75 #taken from [3]
eta = 2.36*10**(-4) #taken from [3]
V1 = 1.18 #taken from [3]
sigma = 5.67*10**(-8)*(86400) #taken from [4] with conversion from seconds to days


# Parameters for open-loop model:
n_q = 5000 
n_h = 10000
a_41 = 0.2
a_12 = 0.5/n_q
a_13 = 0.5/n_h
a_14 = 0.1
a_42 = 0.5/n_q
a_22 = 0.3
a_43 = 0.5/n_h 
a_33 = 0.6
u = 0 #case of no removal of CO2


# Initial conditions:
x01 = 915.4 #as indicated in the source paper [3]
x02 = 210.0 #as indicated in the source paper [3]
x03 = 500.0 #as indicated in the source paper [3]
x04 = 1830.8 #as indicated in the source paper [3]
x05 = 288.33 #as indicated in the source paper [3]

x1e = 637.2 #condition of pre-industrial era
x2e = 0.0
x3e = 0.0
x4e = 1274.4
Te = 287.05 #Temperature corresponding to xe

x_tilde_01 = x01 - x1e  
x_tilde_02 = x02 - x2e
x_tilde_03 = x03 - x3e
x_tilde_04 = x04 - x4e
x_tilde_05 = x05 - Te
x_tilde_0 = np.array([x_tilde_01,
              x_tilde_02,
              x_tilde_03,
              x_tilde_04,
              x_tilde_05])
#################################

# Equations in state-space form:
def ClosedLoopWithTemperature(x_tilde, t=0):
    epsilon = epsilon_n - eta*((x_tilde[0]+x1e)/V1) #translated state in equation of temperature  
    
    return np.array([aHat_11*x_tilde[0] + aHat_12*x_tilde[1] + aHat_13*x_tilde[2] + aHat_14*x_tilde[3],
                     aHat_21*x_tilde[0] + aHat_22*x_tilde[1] + aHat_23*x_tilde[2] + aHat_24*x_tilde[3],
                     aHat_31*x_tilde[0] + aHat_32*x_tilde[1] + aHat_33*x_tilde[2] + aHat_34*x_tilde[3],
                     aHat_41*x_tilde[0] + aHat_42*x_tilde[1] + aHat_43*x_tilde[2] + aHat_44*x_tilde[3],
                     (1/C)*(S*(1-alpha) - epsilon*sigma*(x_tilde[4]+Te)**4)]) #translated state in equation of temperature


def OpenLoopWithTemperature(x_tilde2, t=0):
    epsilon = epsilon_n - eta*((x_tilde2[0]+x1e)/V1) #translated state in equation of temperature  
    
    return np.array([-a_41*x_tilde2[0] + a_14*x_tilde2[3] + n_h*a_13*x_tilde2[2] + n_q*a_12*x_tilde2[1] - u,
                     a_22*x_tilde2[1] - n_q*a_12*x_tilde2[1] - n_q*a_42*x_tilde2[1],
                     a_33*x_tilde2[2] - n_h*a_13*x_tilde2[2] - n_h*a_43*x_tilde2[2],
                     a_41*x_tilde2[0] - a_14*x_tilde2[3] + n_q*a_42*x_tilde2[1] + n_h*a_43*x_tilde2[2],
                     (1/C)*(S*(1-alpha) - epsilon*sigma*(x_tilde2[4]+Te)**4)]) #translated state in equation of temperature)


# Numerical solutions: 
# (1) Closed-loop system:
t_final = 10**4 + 5000 
t = np.linspace(0, t_final, 10**5)
x_tilde, infodict = integrate.odeint(ClosedLoopWithTemperature, x_tilde_0, t,
mxstep = 10**9, full_output = True)
x1_tilde, x2_tilde, x3_tilde, x4_tilde, x5_tilde = x_tilde.T

# (2) Open-loop system:
x_tilde2, infodict2 = integrate.odeint(OpenLoopWithTemperature, x_tilde_0, t,
mxstep = 10**9, full_output = True)
x1_tilde2, x2_tilde2, x3_tilde2, x4_tilde2, x5_tilde2 = x_tilde2.T

epsilon = epsilon_n - eta*((x1_tilde+x1e)/V1)


### Plots:
# Original states, closed loop:
fig = plt.figure(figsize=(10, 10))
plt.plot(t, x1_tilde + x1e, 'b-', label = r'$x_1$', linewidth=6)
plt.plot(t, x2_tilde + x2e, color='orange', label = r'$x_2$', linewidth=6)
plt.plot(t, x3_tilde + x3e, 'g-', label = r'$x_3$', linewidth=6)
plt.plot(t, x4_tilde + x4e, 'r-', label = r'$x_4$', linewidth=6)
plt.grid()
plt.legend(loc='best', prop={'size': 27})
plt.xlabel(r"Time, $t$ [d]", fontsize=35)
plt.ylabel(r"Original state", fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlim(0, 75)

# Original states, open loop:
fig = plt.figure(figsize=(10, 10))
plt.plot(t, x1_tilde2 + x1e, 'b-', label = r'$x_1$', linewidth=6)
plt.plot(t, x2_tilde2 + x2e, color='orange', label = r'$x_2$', linewidth=6)
plt.plot(t, x3_tilde2 + x3e, 'g-', label = r'$x_3$', linewidth=6)
plt.plot(t, x4_tilde2 + x4e, 'r-', label = r'$x_4$', linewidth=6)
plt.grid()
plt.legend(loc='best', prop={'size': 27})
plt.xlabel(r"Time, $t$ [d]", fontsize=35)
plt.ylabel(r"Original state", fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlim(0, 75)

# Temperature, T(t):
if chosen_gain == 'K_of':
    gain_label = 'output feedback'
else:
    gain_label = 'full-state feed.'    
fig = plt.figure(figsize=(10, 10))
plt.plot(t, (x5_tilde + Te) - 273.15, 'k-', linewidth=6, label='Closed (' + gain_label + ')')
plt.plot(t, (x5_tilde2 + Te) - 273.15, 'r-', linewidth=6, label='Open (u = 0)')
plt.grid()
plt.xlabel(r"Time, $t$ [d]", fontsize=40)
plt.ylabel(r"Troposph. temp., $T(t)$ [°C]", fontsize=40) 
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.xlim(0, 10000 + 5000)
plt.legend(loc='upper left', prop={'size': 27})
plt.xticks([0, 5000, 10000, 10**4 + 5000])

# epsilon(t), closed loop:
fig = plt.figure(figsize=(10, 10))
plt.plot(t, epsilon, 'm-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ [d]", fontsize=40)
plt.ylabel(r"Emissivity, $\epsilon(t)$", fontsize=40) 
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.xlim(0, 75)

