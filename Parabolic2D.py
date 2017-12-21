import numpy as np
from math import pi
import matplotlib.pyplot as plt
from tqdm import tqdm

# set dimensions of spatial domain
Lx=2.0 
Ly=1.0

# set Dirichlet boundary conditions
def fT(x):
    # y=0 boundary condition
    u = np.zeros(x.size)
    return u

def fB(x):
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


###############################################################################
    

def Parabolic_2D(maxerr, maxcount, mx, my, mt, lmbdax, lmbday, diags, plot):
    """
    solves the PDE using the conditionally stable 2D forward Euler method
    
    inputs:         maxerr - maximum error between iterations
                    maxcount - the maximum number of iterations to perform
                    mx - the number of x gridpoints
                    my - the number of y gridpoints
                    lambdax - the x mesh number
                    lambday - the y mesh number
    
    """
    # initialise the iteration
    err = maxerr+1
    count = 1    
    
    
    
    # solve the PDE
    for k in tqdm(range(mt)):
        if lmbdax + lmbday > 1/8:
            print('This method is unstable for these values of lambda')
            return 0, 0, 0
        
        for j in range(1,my):
            for i in range(1,mx):
                
                u_new[i, j] = u_old[i, j] + lmbdax* ( u_old[i+1, j] - 2*u_old[i,j] + u_old[i-1, j]) \
                            +               lmbday* ( u_old[i, j+1] - 2*u_old[i,j] + u_old[i, j-1])
            
        err = np.max(np.abs(u_new-u_old))
        u_old[:] = u_new[:]
        count=count+1
        
    if diags == True:    
        # diagnostic output
        print('Final iteration error =',err)
        print('Iterations =',count)
    if plot == True:
        
        # and plot the resulting solution
        xx = np.append(x,x[-1]+deltax)-0.5*deltax  # cell corners needed for pcolormesh
        yy = np.append(y,y[-1]+deltay)-0.5*deltay
        plt.pcolormesh(xx,yy,u_new.T)
        cb = plt.colorbar()
        cb.set_label('u(x,y)')
        plt.xlabel('x'); plt.ylabel('y')
        plt.show()
    return u_new, err, count



###############################################################################
# set numerical parameters

maxerr = 1e-4        # target error
maxcount = 1000      # maximum number of iteration steps

mx = 20
my = 20
mt = 10000

T = 2.0
kappa = 1

# set up the numerical environment variables
x = np.linspace(0, Lx, mx+1)     # mesh points in x
y = np.linspace(0, Ly, my+1)     # mesh points in y
t = np.linspace(0, T, mt+1)
deltax = x[1] - x[0]             # gridspacing in x
deltay = y[1] - y[0]             # gridspacing in y
deltat = t[1] - t[0]
lmbdax = kappa*(deltat/deltax**2)
lmbday = kappa*(deltat/deltay**2)

# set up the solution variables
u_old = np.zeros((x.size,y.size))   # u at current time step
u_new = np.zeros((x.size,y.size))   # u at next time step
u_true = np.zeros((x.size,y.size))  # exact solution


# intialise the boundary conditions, for both timesteps
u_old[1:-1,0] = fB(x[1:-1])
u_old[1:-1,-1] = fT(x[1:-1])
u_old[0,1:-1] = fL(y[1:-1])
u_old[-1,1:-1] = fR(y[1:-1])
u_new[:]=u_old[:]

###############################################################################
###############################################################################


#u_new, err, count = Parabolic_2D(maxerr, maxcount, mx, my, mt, lmbdax, lmbday, True, True)
    
if __name__ == '__main__':
    u_new, err, count = Parabolic_2D(maxerr, maxcount, mx, my, mt, lmbdax, lmbday, True, True)

        
        
        
        






















































