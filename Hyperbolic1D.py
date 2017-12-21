# simple forward Euler solver for the 1D wave equation
#   u_tt = c^2 u_xx  0<x<L, 0<t<T
# with zero-displacement boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial displacement and velocity
#   u=u_I(x), u_t=v_I(x)  0<=x<=L,t=0

import numpy as np
import pylab as plt
from math import pi
import scipy as sp
import matplotlib.animation as animation
from tqdm import tqdm

############### Initial conditions and similar set up #########################
def u_I(x):
    # initial displacement
    y = np.sin(pi*x/L)
    return y

def v_I(x):
    # initial velocity
    y = np.zeros(x.size)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.cos(pi*c*t/L)*np.sin(pi*x/L)
    return y

####################Evolution matrices ########################################

# Explicit finite difference evolution matrix. Tridiag( lmbda^2, 2 - 2lmbda^2, lmbda^2 )
def generate_A_EW(lmbda, mx):
    diagonals_FE = [ (lmbda**2)*np.ones(mx+1)[0:-1], (2- 2*(lmbda**2)*np.ones(mx+1)), (lmbda**2)*np.ones(mx+1)[1:]]
    A_FE = sp.sparse.diags(diagonals_FE, [-1, 0, 1], format='csr')
    A_FE = A_FE.toarray()
    return A_FE

# Implicit LHS finite difference evolution matrix. Tridiag( -0.5 * lmbda^2, 1 + lmbda^2, -0.5 * lmbda^2 )
def generate_A_IW(lmbda, mx):
    diagonals_IW = [ (-0.5*lmbda**2)*np.ones(mx+1)[0:-1], (1 + lmbda**2)*np.ones(mx+1), (-0.5*lmbda**2)*np.ones(mx+1)[1:]]
    A_IW = sp.sparse.diags(diagonals_IW, [-1, 0, 1], format='csr')
    A_IW = A_IW.toarray()
    return A_IW

# Implicit RHS finite difference evolution matrix. Tridiag( 0.5*lmbda^2, -1 - lmbda^2, 0.5 * lmbda^2 )
def generate_B_IW(lmbda, mx):
    diagonals_IW = [ (0.5*lmbda**2)*np.ones(mx+1)[0:-1], (-1 -  lmbda**2)*np.ones(mx+1), (0.5*lmbda**2)*np.ones(mx+1)[1:]]
    B_IW = sp.sparse.diags(diagonals_IW, [-1, 0, 1], format='csr')
    B_IW = B_IW.toarray()
    return B_IW

###############################################################################
    
def plotter(x, u_j, method):
    plt.plot(x,u_j,'r-',label='numeric')
    plt.plot(x,u_exact(x,T),'g-',label='exact') # uncomment to view the exact solution
    plt.xlabel('x')
    plt.ylabel('u(x,T)')
    plt.legend()
    plt.title(method)
    plt.show

##################### Boundary conditions #####################################
def rbound(t):
    return 0


def lbound(t):
    return 0

def BCs(t, x, conditions):
    """
    Defines the boundary conditions depending on their type
    
    inputs:         t - the time for which the conditions are being applied
                    x - the vector of x coordinates on the grid
                    conditions - the type of Boundary Conditions ('Neumann' or 'Dirichlet' (Default))
                    
    outputs:        BC - a vector containing the Boundary conditions at the point asked for
    """
    BC = np.zeros(x.size)

    if conditions.upper() == 'NEUMANN': # not yet implemented
        pass
    
    else:
        BC[1] = lbound(t)
        BC[-2] = rbound(t)
        
    return BC

def q(x):
    """
    Implementation of variable wavespeed. To not use:
        
        return np.ones(x.size)
    
    inputs:         x - position in space of the wave
    
    outputs:        c - change in wavespeed 
    """
    h0 = 1
    a1 = 1
    sigma = 0.005
    x1 = 0
    
    return np.ones(x.size)  # h0 + a1*np.exp( - (( x - x1 )**2)/sigma**2   )
    
###############################################################################
    



def Finite_Difference_EW(u_I, v_I, mx, mt, rbound, lbound, conditions, method, plot, T):
    """
    Explicit 1D finite difference method for solving Hyperbolic equations such as the Wave Equation
    
    inputs:         u_I - the initial displacement conditions
                    v_I - the initial velocity condition
                    mx - the number of x gridpoints
                    mt - the number of t gridpoints
                    rbound - the right hand boundary condition
                    lbound - the left hand boundary condition
                    conditions - the type of boundary condition ('Neumann', 'Dirichlet' (Default))
                    T - the length of time for which to solve
                    
    outputs:        u_jp1 - the solution of the PDE after the prescribed period of time
    
    example:        U = Finite_Difference_EW(u_I, v_I, i, mt, rbound, lbound, '', T)
                    a plot of the solution of the PDE after the prescribed time

    """
    
    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # gridpoints in space
    t = np.linspace(0, T, mt+1)     # gridpoints in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lambdasqr = (c*deltat/deltax)**2    # squared courant number
    lmbda = (c*deltat/deltax)    # courant number
    
    #print("lambda=",np.sqrt(lambdasqr))


    A_EW = generate_A_EW(lmbda, mx)
    
    # set up the solution variables
    u_jm1 = np.zeros(x.size)        # u at previous time step
    u_j = np.zeros(x.size)          # u at current time step
    u_jp1 = np.zeros(x.size)        # u at next time step

    # Set initial condition
    u_jm1 = u_I(x)    

    # First timestep              
    u_j[1:-1] = u_jm1[1:-1] + 0.5*A_EW[1:-1, 1:-1].dot(u_I(x)[1:-1]) + \
                deltat*v_I(x)[1:-1]


    # First timestep boundary condition
    u_j += lmbda*BCs(0, x, conditions)

    for n in range(2, mt+1):



        # regular timestep at inner mesh points
        u_jp1[1:-1] = A_EW[1:-1, 1:-1].dot(u_j[1:-1]) - u_jm1[1:-1]

        # boundary conditions
        u_jp1 += (lmbda**2) *BCs(n/mt, x, conditions)

        # update u_jm1 and u_j
        u_jm1[:],u_j[:] = u_j[:],u_jp1[:]

    # 'Bolt on' the boundary condtions for the graph
    u_jp1[0] = lbound(T)
    u_jp1[-1] = rbound(T)
    
    if plot == True:
        plotter( x, u_jp1, method)
        
    # Plot the final result and exact solution


    return u_jp1

def Finite_Difference_IW(u_I, v_I, mx, mt, rbound, lbound, conditions, method, plot, T):
    """
    Implicit 1D finite difference method for solving Hyperbolic equations such as the Wave Equation
    
    inputs:         u_I - the initial displacement conditions
                    v_I - the initial velocity condition
                    mx - the number of x gridpoints
                    mt - the number of t gridpoints
                    rbound - the right hand boundary condition
                    lbound - the left hand boundary condition
                    conditions - the type of boundary condition ('Neumann', 'Dirichlet' (Default))
                    T - the length of time for which to solve
                    
    outputs:        u_jp1 - the solution of the PDE after the prescribed period of time
    
    example:        U = Finite_Difference_EW(u_I, v_I, i, mt, rbound, lbound, '', T)
                    a plot of the solution of the PDE after the prescribed time
    """
    
    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # gridpoints in space
    t = np.linspace(0, T, mt+1)     # gridpoints in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lambdasqr = (c*deltat/deltax)**2    # squared courant number
    lmbda = (c*deltat/deltax)    # courant number
    
    #print("lambda=",np.sqrt(lambdasqr))


    A_IW = generate_A_IW(lmbda, mx)
    B_IW = generate_B_IW(lmbda, mx)
    
    # set up the solution variables
    u_jm1 = np.zeros(x.size)        # u at previous time step
    u_j = np.zeros(x.size)          # u at current time step
    u_jp1 = np.zeros(x.size)        # u at next time step
    qx = q(x)
    
    # Set initial condition
    u_jm1 = u_I(x)    
 
    # First timestep              
    u_j[1:-1] = sp.sparse.linalg.spsolve( A_IW[1:-1, 1:-1] - B_IW[1:-1, 1:-1] ,\
                                           2*(u_I(x)[1:-1] - B_IW[1:-1, 1:-1] \
                                              .dot( deltat * v_I(x)[1:-1])))

    # First timestep boundary condition
    u_j += lmbda*BCs(0, x, conditions)

    for n in range(2, mt+1):

        # boundary conditions
        u_j += (lmbda**2) *BCs(n/mt, x, conditions)

        # regular timestep at inner mesh points
        u_jp1[1:-1]    = qx[1:-1] * sp.sparse.linalg.spsolve( A_IW[1:-1,1:-1], (2*u_j[1:-1] \
                                         + B_IW[1:-1,1:-1].dot(u_jm1[1:-1])) )
        
        # update u_jm1 and u_j
        u_jm1[:],u_j[:] = u_j[:],u_jp1[:]

    # Plot the final result and exact solution
    u_jp1[0] = lbound(T)
    u_jp1[-1] = rbound(T)
    
    # Plot the final result and exact solution

    if plot == True:
        plotter( x, u_jp1, method)


    return u_jp1


###############################################################################
def Compute_Error(x, y):
    """
    Calculates the max absolute value between two input arguments
    
    inputs:             x - a number/vector/matrix
                        y - a number/vector/matrix
    
    output:             e - the max absolute value vetween x and y
    """
    e = abs(x - y)
    e = max(e)
    return e
    
    
def ErrorVn(max_tries, method, d):
    """
    Demonstrates the error for the different schemes mentioned above. It investivates 
    the effect of changing the step size in time, or space, depending on the inputs
    
    inputs:             max_tries - the maximum number of iterations to run the code for (6 minimum)
                        method - the name of the scheme to investigate ('FE', 'BE', 'CN')
                        d - the grid space to vary ('dx', 'dt' )
                        
    outputs:            e - the vector of errors from the true solution at every point
                        a plot of the log(error) varying against stepsize
    """

    e = np.zeros(max_tries-5)
    x_val = np.zeros(max_tries-5)
    if method.upper() == 'EXPLICIT':
        if d.upper() == 'DX':
            for i in tqdm(range(5, max_tries)):
                
                x = np.linspace(0, L, (1*i)+1)
                exact_soln = u_exact(x, T)
                approx_soln = Finite_Difference_EW(u_I, v_I, i, mt, rbound, lbound, '', 'Explicit', False, T)
                e[i-5] = Compute_Error(exact_soln, approx_soln)
                
                x_val[i-5] = 1 / i
                
        if d.upper() == 'DT':   
           for i in tqdm(range(5, max_tries)):
                
               x = np.linspace(0, L, (1*i)+1)
               exact_soln = u_exact(x, T)
               approx_soln = Finite_Difference_EW(u_I, v_I, mx, i, rbound, lbound, '', 'Explicit', False ,T)
               e[i-5] = Compute_Error(exact_soln, approx_soln)
                
               x_val[i-5] = 1 / i
               
    if method.upper() == 'IMPLICIT':
        if d.upper() == 'DX':
            for i in tqdm(range(5, max_tries)):
                
                x = np.linspace(0, L, (1*i)+1)
                exact_soln = u_exact(x, T)
                approx_soln = Finite_Difference_IW(u_I, v_I, i, mt, rbound, lbound, '', 'Implicit', False, T)
                e[i-5] = Compute_Error(exact_soln, approx_soln)
                
                x_val[i-5] = 1 / i
                
        if d.upper() == 'DT':   
           for i in tqdm(range(5, max_tries)):
                
               x = np.linspace(0, L, (1*i)+1)
               exact_soln = u_exact(x, T)
               approx_soln = Finite_Difference_IW(u_I, v_I, mx, i, rbound, lbound, '', 'Implicit', False, T)
               e[i-5] = Compute_Error(exact_soln, approx_soln)
                
               x_val[i-5] = 1 / i
    plt.plot(x_val, np.log(e)  )
    plt.show()
        
    return e, x_val

###################### Animations #############################################
def animate(animtime, method):
    """
    An attempt to animate the solution to the wave equation as it evolves over time
    However two problems arose:
                                Spyder refuses to behave like a GUI
                                It is impossible to save the animation on the computer used to develop this software
                                
    At present, it seems to produce multiple lines showing the solution evaluated at different time points
    
    inputs:             animtime - the number of seconds to animate for
                        method - the method you would like to use for your animations
    
    """
    fig = plt.figure()
    plts = []
    plt.hold("off")
    
    if method.upper() == 'IMPLICIT':
        for i in range(animtime):
            p = plt.plot(Finite_Difference_IW(u_I, v_I, mx, mt, rbound, lbound, '', i/4))
            plts.append(p)
    
    else:
        for i in range(animtime):
            p = plt.plot(Finite_Difference_IW(u_I, v_I, mx, mt, rbound, lbound, '', i/4))
            plts.append(p)
                
    ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)   # run the animation
    ani.save('wave.mp4')  #uncomment to try to save the animation. Requires ffmpeg or similar
    fig.show()

###############################################################################
###############################################################################

# Set problem parameters/functions
c=1.0         # wavespeed
L=1.0         # length of spatial domain
T=2.0         # total time to solve for

# Set numerical parameters
mx = 300     # number of gridpoints in space
mt = 600     # number of gridpoints in time

#e, x_val = ErrorVn(200, 'implicit', 'dx')
#animate(3, 'implicit')
#Finite_Difference_IW(u_I, v_I, mx, mt, rbound, lbound, '', 'Implicit', True, T)

if __name__ == '__main__':
    Finite_Difference_EW(u_I, v_I, mx, mt, rbound, lbound, '', 'Explicit', True, T)