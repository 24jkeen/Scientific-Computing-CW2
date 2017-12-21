# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:57:52 2017

@author: 24jke
"""

# finite difference solver (Jacobi iteration) for the 2D Laplace equation
#   u_xx + u_yy = 0  0<x<Lx, 0<y<Ly
# with Dirichlet boundary conditions
#   u = fB(x) 0<=x<=Lx, y=0
#   u = fT(x) 0<=x<=Lx, y=Ly
#   u = fL(y) x=0, 0<=y<=Ly
#   u = fR(y) x=Lx, 0<=y<=Ly


import numpy as np
#import pylab as pl
from math import pi
import matplotlib.pyplot as plt


###############################################################################

# set Dirichlet boundary conditions
def fB(x):
    # y=0 boundary condition
    u = np.zeros(x.size)
    return u

def fT(x):
    # y=Ly boundary condition
    u = np.sin(pi*x/Lx)
    return u

def fL(y):
    # x=0 boundary condition
    u = np.zeros(y.size)
    return u

def fR(y):
    # x=Lx boundary condition
    u = np.zeros(y.size)
    return u

def u_exact(x,y):
    # the exact solution
    y = np.sin(pi*x/Lx)*np.sinh(pi*y/Lx)/np.sinh(pi*Ly/Lx)
    return y

###############################################################################
    
def Elliptic_Solver(maxerr, maxcount, mx, my, Lx, Ly, w , diags, plot):
    """
    A solver that relies on Gauss Seidel and Successive Over Relaxation to solve 
    Hyperbolic PDE's in 2D
    
    inputs:         maxerr - the maximum error between approximations
                    maxcount - the maximum number of iterations before the scheme gives up
                    mx - number of x gridpoints
                    my - number of y gridpoints
                    Lx - the x dimension of the spatial domain
                    Ly - the y dimension of the spatial domain
                    w - the coefficient for SOR (1 < w < 2). if 1 then simply Gauss Seidel

    outputs:        u_new - solution at the final approximation
                    err - the error between the last approximation and the one before
                    count - the number of iterations to arrive at the solution
                    u_true - the actual solution
    """
    # initialise the iteration
    err = maxerr+1
    count = 1
    
    # set up the numerical environment variables
    x = np.linspace(0, Lx, mx+1)     # mesh points in x
    y = np.linspace(0, Ly, my+1)     # mesh points in y
    deltax = x[1] - x[0]             # gridspacing in x
    deltay = y[1] - y[0]             # gridspacing in y
    lambdasqr = (deltax/deltay)**2       # mesh number
    
    # set up the solution variables
    u_old = np.zeros((x.size,y.size))   # u at current time step
    u_new = np.zeros((x.size,y.size))   # u at next time step
    R = np.ones((x.size,y.size))  
    
    
    # true solution values on the grid
    
    u_true = np.zeros((x.size,y.size))  # exact solution
    for i in range(0,mx+1):
        for j in range(0,my+1):
            u_true[i,j] = u_exact(x[i],y[j])
    
    # intialise the boundary conditions, for both timesteps
    u_old[1:-1,0] = fB(x[1:-1])
    u_old[1:-1,-1] = fT(x[1:-1])
    u_old[0,1:-1] = fL(y[1:-1])
    u_old[-1,1:-1] = fR(y[1:-1])
    u_new[:]=u_old[:]
    
    # solve the PDE
    while err>maxerr and count<maxcount:
        for j in range(1,my):
            for i in range(1,mx):
                
            
                #Define the residual
                R[i, j] = ( u_new[i-1,j] + u_old[i+1,j] - 2*(1 + lambdasqr )*u_old[i,j] + \
                              lambdasqr*(u_old[i,j+1]+u_new[i,j-1]) )/(2*(1+lambdasqr))            
            
                # Update the approximation for every gridpoint
                u_new[i,j] = u_old[i,j]  + w * R[i,j]
        
                
                #print(R[i, j])
        #print(R)
        err = np.max(np.abs(u_new-u_old))
        u_old[:] = u_new[:]
        count=count+1
        
        
    if diags == True:
        # calculate the error, compared to the true solution    
        err_true = np.max(np.abs(u_new[1:-1,1:-1]-u_true[1:-1,1:-1]))
        # diagnostic output
        print('Final iteration error =',err)
        print('Iterations =',count)
        print('Max diff from true solution =',err_true)
        
    if plot == True:
        # and plot the resulting solution

        xx = np.append(x,x[-1]+deltax)-0.5*deltax  # cell corners needed for pcolormesh
        yy = np.append(y,y[-1]+deltay)-0.5*deltay
        plt.pcolormesh(xx,yy,u_new.T)
        cb = plt.colorbar()
        cb.set_label('u(x,y)')
        plt.xlabel('x'); plt.ylabel('y')
        plt.show()
            
            
    return u_new

###############################################################################
###############################################################################
    
# set dimensions of spatial domain
Lx=2.0 
Ly=1.0

# set numerical parameters
mx = 40              # number of gridpoints in x
my = 20              # number of gridpoints in y
maxerr = 1e-4        # target error
maxcount = 1000      # maximum number of iteration steps
w = 1.85


#u_new = Elliptic_Solver(maxerr, maxcount, mx, my, Lx, Ly, w, True, True )

if __name__ == '__main__':   
    u_new = Elliptic_Solver(maxerr, maxcount, mx, my, Lx, Ly, w, True, True )
    