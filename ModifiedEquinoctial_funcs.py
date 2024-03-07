# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:11:35 2023

@author: sina
"""
# Modified Equinoctial Orbital Elements Functions


import numpy as np
import math

def kep2meoe(kep):
    kep = kep.reshape((6,))
    a = kep[0]
    e = kep[1]
    i = kep[2]
    raan = kep[3]
    aop = kep[4]
    theta = kep[-1]
    
    p = a*(1-e**2)
    f = e*math.cos(aop+raan)
    g = e*math.sin(aop+raan)
    h = math.tan(i/2)*math.cos(raan)
    k = math.tan(i/2)*math.sin(raan)
    L = raan + aop + theta
    
    meoe = np.array([p,f,g,h,k,L])
    
    return meoe

def meoe2kep(meoe):
    [p,f,g,h,k,L]=meoe
    a = p/(1-f**2-g**2)
    e = (f**2+g**2)**0.5
    i = 2*math.atan((h**2+k**2)**0.5)
    if e == 0:
        aop=0
    else:
        #aop = math.atan(g/f)-math.atan(k/h)
        aop = math.atan2(g*h-f*k, f*h+g*k)
    if i == 0:
        raan = 0
    else:
        raan = math.atan2(k,h) # 4 Quadrant inverse tangent
    theta = L - (raan+aop)
    kep = np.array([a,e,i,aop,raan,theta])
    return kep


def d_meoe (meoe,t,*args): # args can be (pert_acc,mu) or just the acceleration vector
    [p,f,g,h,k,L]=meoe
    if len(args)<2:
        mu = 398600.44
        a = args[0]
    else:
        mu = args[1]
        a = args[0]
        
    [ar,at,an]=a
    w = 1+f*math.cos(L)+g*math.sin(L)
    s2 = 1+h**2+k**2
    dp = 2*(p/w)*(p/mu)**0.5 * at
    df = (p/mu)**0.5 * (ar*math.sin(L)+((w+1)*math.cos(L)+f)*at/w-(h*math.sin(L)-k*math.cos(L))*g*an/w)
    dg = (p/mu)**0.5 * (-ar*math.cos(L)+((w+1)*math.sin(L)+g)*at/w+(h*math.sin(L)-k*math.cos(L))*g*an/w)
    dh = (p/mu)**0.5 * s2*an*math.cos(L)/(2*w)
    dk = (p/mu)**0.5 * s2*an*math.sin(L)/(2*w)
    dL = (mu*p)**0.5 * (w/p)**2 + (p/mu)**0.5 *(h*math.sin(L)-k*math.cos(L))*an/w
    dmeoe = np.array([dp,df,dg,dh,dk,dL])
    return dmeoe

def meoe2car(meoe,*mu):
    if len(mu)<1:
        mu = 398600.44
    [p,f,g,h,k,L]=meoe
    alpha2 = h**2-k**2
    s2=1+h**2+k**2
    w = 1+f*math.cos(L)+g*math.sin(L)
    r=p/w
    
    r1 = r/s2*(math.cos(L)+alpha2*math.cos(L)+2*h*k*math.sin(L))
    r2 = r/s2*(math.sin(L)-alpha2*math.sin(L)+2*h*k*math.cos(L))
    r3 = 2*r/s2*(h*math.sin(L)-k*math.cos(L))
    
    r = np.array([r1,r2,r3])
    
    v1 = -1/s2*(mu/p)**0.5*(math.sin(L)+alpha2*math.sin(L)-2*h*k*math.cos(L)+g-2*f*h*k+alpha2*g)
    v2 = -1/s2*(mu/p)**0.5*(-math.cos(L)+alpha2*math.cos(L)+2*h*k*math.sin(L)-f+2*g*h*k+alpha2*f)
    v3 = 2/s2*(mu/p)**0.5*(h*math.cos(L)+k*math.sin(L)+f*h+g*k)
    
    v = np.array([v1,v2,v3])
    
    car = np.stack([r,v]).reshape((6,))
    
    return car