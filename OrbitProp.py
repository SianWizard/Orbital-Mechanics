# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:32:18 2024

@author: sina
"""

# Orbit propagation functions

import numpy as np
from scipy.integrate import odeint
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def dOrb(X,t=0,miu=398600.44,a=np.zeros((3,))):
    X=X.reshape(-1)
    r = X[0:3]
    dr = X[3:6]
    ddr = -miu*r/(np.linalg.norm(r))**3 + a
    dx = np.concatenate((dr,ddr),axis=0)
    return dx

def OrbitProp_noPert (X0,tf_sec,dt_sec,miu = 398600.44):
    t = np.arange(0,tf_sec,dt_sec)
    X = odeint(dOrb,X0,t,rtol=1e-12,atol=1e-12)
    return X

def dispOrb (X):
    if X.shape[1] != 6:
        X = X.T
    pos = X[:,0:3]
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(pos[:,0],pos[:,1],pos[:,2])
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.axis('equal')

def inertial2body (X):
    X = X.reshape(-1)
    r = X[0:3]
    dr = X[3:6]
    RotMat = np.array([r,dr,np.cross(r,dr)]).T
    return RotMat


def OrbFunc_Test ():
    x0 = np.array([7000,0,0,0,8,0])
    tf = 1*24*3600
    dt = 5
    x = OrbitProp_noPert(x0,tf,dt)
    #dispOrb(x)
    
    # Just checking the effect of cross-track difference
    x0_2 = x0+np.array([0,0,0.001,0,0,0])
    x_2 = OrbitProp_noPert(x0_2,tf,dt)
    x_diff = x_2-x
    dispOrb(x_diff)
    dispOrb(x)