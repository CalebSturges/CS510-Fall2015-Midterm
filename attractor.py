#!/usr/bin/python
# Filename: attractor.py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Attractor:
    """Numerically analizes a differntial equation using different integration methods.
    
    Keyword arguments:
    S -- Constant in differntial equation (default 10.)
    B -- Constant in differntial equation (default 8./3.)
    P -- Constant in differntial equation (default 28.)
    """
    def __init__(self, S = 10., B = 8./3., P = 28.):
        self.s = S
        self.b = B
        self.p = P
        self.params = np.array([S,B,P])
        self.start = 0.
        self.end = 80.
        self.points = 10000
        self.dt = (self.end-self.start)/self.points
        
    def dxdt(self, npArray):
        """Computes dxdt of differntial equations, takes numpy array of the the three parameters"""
        return self.s*(npArray[1]-npArray[0])
    
    def dydt(self, npArray):
        """Computes dydt of differntial equations, takes numpy array of the the three parameters"""
        return npArray[0]*(self.p-npArray[2])-npArray[1]
    
    def dzdt(self, npArray):
        """Computes dzdt of differntial equations, takes numpy array of the the three parameters"""
        return npArray[0]*npArray[1]-self.b*npArray[2]
    
    def euler(self, npArray):
        """Computes differntial equation using eulers method, takes numpy array of the the three parameters"""
        return npArray + np.array([self.dxdt(npArray),self.dydt(npArray), self.dzdt(npArray)])*self.dt 
    
    def rk2(self, npArray):
        """Computes differntial equation using runge-kutta 2 method, takes numpy array of the the three parameters"""
        k1 =  np.array([self.dxdt(npArray),self.dydt(npArray), self.dzdt(npArray)])
        n1step = npArray + k1*self.dt/2
        k2 =  np.array([self.dxdt(n1step),self.dydt(n1step), self.dzdt(n1step)])
        return npArray + k2*self.dt

    def rk4(self, npArray):
        """Computes differntial equation using runge-kutta 4 method, takes numpy array of the the three parameters"""
        k1 =  np.array([self.dxdt(npArray), self.dydt(npArray), self.dzdt(npArray)])
        n1step = npArray + k1*self.dt/2
        k2 =  np.array([self.dxdt(n1step), self.dydt(n1step), self.dzdt(n1step)])
        n1step = npArray + k2*self.dt/2
        k3 =  np.array([self.dxdt(n1step), self.dydt(n1step), self.dzdt(n1step)])
        n1step = npArray + k3*self.dt
        k4 =  np.array([self.dxdt(n1step), self.dydt(n1step), self.dzdt(n1step)])
        return npArray + (k1+2*k2+2*k3+k4)/6
    
    def evolve(self, r0 = np.array([0.1,0.,0.]), order = 4):
        """Numerically analizes differntial equation saves result, takes numpy array of the the three parameters and order of integration."""
        if order==1:
            orderFunc=self.euler
        elif order==2:
            orderFunc=self.rk2
        else:
            orderFunc=self.rk2
        
        myList=[]
        myList.append(np.array([.1,.0,.0]))
        for i in np.arange(1, self.points):
            myList.append(orderFunc(myList[i - 1]))
        columns = ['t', 'x', 'y', 'z']
        myArray =np.array([[self.dt*i,myList[i][0],myList[i][1],myList[i][2]] for i in np.arange(self.points)])  
        return pd.DataFrame(myArray, columns=columns)
        
    def plotx(self, df):
        """Plots x as a function of t"""
        plt.plot(df['t'], df['x'])
        
    def ploty(self, df):
        """Plots y as a function of t"""
        plt.plot(df['t'], df['y'])
        
    def plotz(self, df):
        """Plots z as a function of t"""
        plt.plot(df['t'], df['z'])

    def plotxy(self, df):
        """Plots x as a function of y"""
        plt.plot(df['x'], df['y'])
        
    def plotyz(self, df):
        """Plots y as a function of z"""
        plt.plot(df['y'], df['z'])
        
    def plotxz(self, df):
        """Plots x as a function of x"""
        plt.plot(df['x'], df['z'])

    def plot3d(self, df):
        """Plots x, y and z for all t"""
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(df['x'], df['y'], df['z'])


