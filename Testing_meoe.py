# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:37:43 2023

@author: sina
"""

import ModifiedEquinoctial_funcs as MF
import numpy as np
import math
from scipy.integrate import odeint

# Plotting
#from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


mu = 398600.44
kep0 = [42000, 0.2 , 1 , 0 , 0 , 0 ]

T = 2*math.pi*(kep0[0]**3/mu)**0.5

kep0 = np.array(kep0)

meoe0 = MF.kep2meoe(kep0)
print("This is the Modified Equinoctial Orbital Elements:")
print(meoe0)

kep_returned0 = MF.meoe2kep(meoe0)

error = kep0 - kep_returned0

print("The error in the conversions back and forth is:")
print(np.linalg.norm(error)) 

time = np.linspace(0, T, 1000)
propagated_meoe = odeint(MF.d_meoe,meoe0,time,args=(np.array([0,0,0]),),rtol = 1e-12,atol = 1e-12)

R = np.zeros((time.size,3))

for i in range(0,propagated_meoe.shape[0]): # or time.size
    temp_car = MF.meoe2car(propagated_meoe[i][:])
    R[i,:] = temp_car[0:3]

x_series = R[:,0]

fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
ax.set_box_aspect([1, 1, 1])
#ax.set_aspect('equal', 'box')

 
# plotting
ax.plot3D(R[:,0], R[:,1], R[:,2], 'green')
ax.set_title('Orbit')
plt.show()

