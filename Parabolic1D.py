
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi

def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-(pi**2)*t)*np.sin(pi*x)
    return y


######## Second order matrices ################################################

# Forward Euler. Tridiag( lmbda, 1-2*lmbda, lmbda )
def generate_A_FE(lmbda, mx):
    diagonals_FE = [ lmbda*np.ones(mx+1)[0:-1], (1- 2*lmbda)*np.ones(mx+1), lmbda*np.ones(mx+1)[1:]]
    A_FE = sp.sparse.diags(diagonals_FE, [-1, 0, 1], format='csc')
    A_FE = A_FE.toarray()
    return A_FE

# Backward Euler. Tridiag( -lmbda, 1+2*lmbda, -lmbda )
def generate_A_BE(lmbda, mx):
    diagonals_BE = [ -lmbda*np.ones(mx+1)[0:-1], (1+ 2*lmbda)*np.ones(mx+1), -lmbda*np.ones(mx+1)[1:]]
    A_BE = sp.sparse.diags(diagonals_BE, [-1, 0, 1], format='csc')
    A_BE = A_BE.toarray()
    return A_BE

# Crank Nicholson LHS. Tridiag( -lmbda/2, 1+lmbda, -lmbda/2 )
def generate_A_CN(lmbda, mx):
    diagonals_CN = [ (-lmbda/2)*np.ones(mx+1)[0:-1], (1+ lmbda)*np.ones(mx+1), (-lmbda/2)*np.ones(mx+1)[1:]]
    A_CN = sp.sparse.diags(diagonals_CN, [-1, 0, 1], format='csc')
    A_CN = A_CN.toarray()
    return A_CN

# Crank Nicholson RHS. Tridiag( lmbda/2, 1-lmbda, lmbda/2 )
def generate_B_CN(lmbda, mx):
    diagonals_CN = [ (lmbda/2)*np.ones(mx+1)[0:-1], (1 - lmbda)*np.ones(mx+1), (lmbda/2)*np.ones(mx+1)[1:]]
    B_CN = sp.sparse.diags(diagonals_CN, [-1, 0, 1], format='csc')
    B_CN = B_CN.toarray()
    return B_CN
    

################Boundary Conditions and RHS functions #########################
def LHBound(t):
    return 0
    
def RHBound(t):
    return 0

def F(x, t):
    val = x**2
    return np.zeros(x.size)

###############################################################################
    
def Forward_Euler(lmbda, u_j, u_jp1, mx, mt, a, b, conditions, x):
    """
    Explicit 1D Forward Euler method for solving Parabolic PDE's. Conditionally stable for 
    
    lmbda = K* (deltat / deltax^2) < 0.5
    
    inputs:         lmbda - the mesh Fourier number (defined above)
                    u_j - the value of the PDE at this moment in time
                    u_jp1 - the value of the PDE at the next moment in time
                    mx - the number of x gridpoints in the mesh
                    mt - the number of time gridpoints in the mesh
                    a - the Left hand boundary condition
                    b - the right hand boundary condition
                    conditions - the type of boundary conditions, Neumann or Dirichlet (Default Dirichlet)
                    x - the gridpoints in the spatial domain along which to solve
                    
    outputs:        u_j - the PDE evaluated at the final time step
    
    example:        u_jFE = Forward_Euler(lmbda, u_j, u_jp1, mx, mt, a, b, conditions, x)
                    
                    >>> u_j 
    
    """
    A_FE = generate_A_FE(lmbda, mx)
    BC = np.zeros_like(u_j)                         #create empty Boundary conditions vector
    deltax = 1/mx
    deltat = 1/mt
    

    if conditions.upper() == 'NEUMANN':             # if the boundary conditions are Neumann
        A_FE[1, 2] = A_FE[1, 2]*2                   # we must change the evolution matrix
        A_FE[-2, -3] = A_FE[-2, -3]*2

    for n in range(0, mt+1):                        # caluculate the value of the next point
                                                    # forward in time on the grid
        
        if conditions.upper() == 'NEUMANN':         # If Neumann conditions the BC's behave differently
            BC[1] = -a(n/mt)
            BC[-2] = b(n/mt)
            
            BC = 2*lmbda*deltax*BC
            
        else:
            BC[1] = a(n/mt)                         # If not Neumann, set Dirichlet BC's
            BC[-2] = b(n/mt)
                        
            
            BC = lmbda*BC
        # Evaluate next point forward in time with a RHS function
        u_jp1[1:-1] = A_FE[1:-1, 1:-1].dot(u_j[1:-1]) + deltat * F(x, n)[1:-1]
        
        # Apply Boundary Conditions
        u_jp1 += BC
    
        # update
        u_j[:] = u_jp1
        
    # 'Bolt on' the boundary conditions for the graph
    u_j[0] = a(T)
    u_j[-1] = b(T)

    return u_j

def Backward_Euler(lmbda, u_jm1, u_j, mx, mt, a, b, conditions, x):
    """
    Implicit 1D Backward Euler method for solving Parabolic PDE's. Unconditionally stable  
    
    lmbda = K* (deltat / deltax^2) < 0.5
    
    inputs:         lmbda - the mesh Fourier number (defined above)
                    u_jm1 - the value of the PDE at the previous moment in time
                    u_j - the value of the PDE at this moment in time
                    mx - the number of x gridpoints in the mesh
                    mt - the number of time gridpoints in the mesh
                    a - the Left hand boundary condition
                    b - the right hand boundary condition
                    conditions - the type of boundary conditions, Neumann or Dirichlet (Default Dirichlet)
                    x - the gridpoints in the spatial domain along which to solve
                    
    outputs:        u_j - the PDE evaluated at the final time step
    
    example:        u_jm1BE = Backward_Euler(lmbda, u_jm1, u_j, mx, mt, a, b, conditions, x)

                    >>> u_jm1
    
    """
    A_BE = generate_A_BE(lmbda, mx)
    BC = np.zeros_like(u_j)                     # create an empty vector for BC's
    deltax = 1/mx
    deltat = 1/mt

    if conditions.upper() == 'NEUMANN':         # if neumann BC's adapt the evolution matrix
        A_BE[1, 2] = A_BE[1, 2]*2
        A_BE[-2, -3] = A_BE[-2, -3]*2
    
    for n in range(0, mt + 1):                  # evaluate at every time point on the grid

        
        if conditions.upper() == 'NEUMANN':
            BC[0] = -a(n/mt)                    # Neumann BC's are applied differently so
            BC[-1] = b(n/mt)                    # are handled sperately
            
            BC = 2*lmbda*deltax*BC
            
        else:                                   # if not neumann BC's then Dirichlet are default
            BC[1] = a(n/mt)
            BC[-2] = b(n/mt)
                        
            BC = lmbda*BC

        
        u_j[1:-1] += deltat * F(x, n)[1:-1]     # apply the right hand side function

        u_j += BC                               # apply the BC's
        
        # Solve a system of equations for u_jm1
        u_jm1[1:-1] = sp.sparse.linalg.spsolve( A_BE[1:-1, 1:-1], u_j[1:-1])

        # update 
        u_j[:] = u_jm1
        
    # 'Bolt on' the boundary conditions for the graph
    u_jm1[0] = a(0)
    u_jm1[-1] = b(T)
        
    return u_jm1
    
def Crank_Nicholson(lmbda, u_jp1, u_j, mx, mt, a, b, conditions, x):
    """
    Implicit 1D Crank Nicholson method for solving Parabolic PDE's. Unconditionally stable  
    
    lmbda = K* (deltat / deltax^2) < 0.5
    
    inputs:         lmbda - the mesh Fourier number (defined above)
                    u_jp1 - the value of the PDE at the next moment in time
                    u_j - the value of the PDE at this moment in time
                    mx - the number of x gridpoints in the mesh
                    mt - the number of time gridpoints in the mesh
                    a - the Left hand boundary condition
                    b - the right hand boundary condition
                    conditions - the type of boundary conditions, Neumann or Dirichlet (Default Dirichlet)
                    x - the gridpoints in the spatial domain along which to solve
                    
    outputs:        u_j - the PDE evaluated at the final time step
    
    example:        u_jp1CN = Crank_Nicholson(lmbda, u_jp1, u_j, mx, mt, a, b, conditions, x)

                    >>> u_j
    
    """
    A_CN = generate_A_CN(lmbda, mx)             # Generate LHS and RHS evolution matrices
    B_CN = generate_B_CN(lmbda, mx)
    BC = np.zeros_like(u_j)                     # create empty BC's vector
    deltax = 1/mx
    deltat = 1/mt
    
    if conditions.upper() == 'NEUMANN':         # if BC's are Neumann the evolution matrix
        A_CN[0, 1] = A_CN[0, 1]*2               # must be changed
        A_CN[-1, -2] = A_CN[-1, -2]*2

        B_CN[0, 1] = B_CN[0, 1]*2
        B_CN[-1, -2] = B_CN[-1, -2]*2
    
    for n in range(mt+1):                       # solve PDE for every time point on the grid
        
        if conditions.upper() == 'NEUMANN':     # neumann BC's are handled seperately here
            BC[0] = -a(n/mt)
            BC[-1] = b(n/mt)
            
            BC = 2*lmbda*deltax*BC
            
        else:                                   # Dirichlet are the default BC's
            BC[1] = a(n/mt)
            BC[-2] = b(n/mt)
                        
            BC = lmbda*BC
            
            
        B = B_CN.dot(u_j) + BC + deltat*F(x, n) #Precompute the Right hand side with BCs and RHS function

        # solve the system of equations for u_jp1
        u_jp1[1:-1] = sp.sparse.linalg.spsolve(A_CN[1:-1, 1:-1], B[1:-1])
        
        #update
        u_j[:] = u_jp1
        
       
    # 'Bolt on' the boundary conditions for the graph
    u_j[0] = a(T)
    u_j[-1] = b(T)
        
    return u_j
    
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
    
    e = np.zeros(max_tries-5)               # less than 5 comparisons is pointless right?
    x_val = np.zeros(max_tries-5)
        
    if d.upper() == 'DX':
        for i in tqdm(range(5, max_tries)):# tqdm makes progress bars
            
            x = np.linspace(0, L, (1*i)+1)  
            exact_soln = u_exact(x, T)      # compute the exact solution to compare against
            
            # approximate the solution to the PDE
            approx_soln = Parabolic_PDE(kappa, L, T, u_I, 1*i, mt, LHBound, RHBound, method, condition, False)
            e[i-5] = Compute_Error(exact_soln, approx_soln)
            
            
            x_val[i-5] = 1 / i  # converts the x axis of the graph to 'size of gridpoint'
            
    if d.upper() == 'DT':   
       for i in tqdm(range(5, max_tries)):
            
           x = np.linspace(0, L, (1*i)+1)
           exact_soln = u_exact(x, T)
           
            # approximate the solution to the PDE
           approx_soln = Parabolic_PDE(kappa, L, T, u_I, mx, 100*i, LHBound, RHBound, method, condition, False)
           e[i-5] = Compute_Error(exact_soln, approx_soln)
            
           x_val[i-5] = 1 / i
    
    plt.xlabel('Step Size')
    plt.ylabel('log(|error|)')
    plt.plot(x_val, np.log(e)  )
    plt.show()
        
    return e, x_val
        

###############################################################################
    
def plotter(x, u_j, method):
    plt.plot(x,u_j,'ro', label='numeric')          
    plt.plot(x,u_exact(x,T),'g-',label='exact') 
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(method.upper())
    plt.show()
    
#################### Main function ############################################
    
def Parabolic_PDE(kappa, L, T, u_I, mx, mt, a, b, method, conditions, plot):
    """
    The main solver function of this file. If you want your parabolic PDE solved
    good and proper, this is the function for you. 
    
    inputs:         kappa - Diffusion coefficient
                    L - length of spatial domain
                    T - time period to solve over
                    u_I - initial conditions
                    mx - number of spacial gridpoints
                    mt - number of time gridpoints
                    a - LHS boundary condition
                    b - RHS boundary condition
                    method - method used to solve ('FE', 'BE', 'CN')
                    conditions - type of boundary conditions ('Neumann', 'Dirichlet' (Default))
                    plot - optional plot True of False
    outputs:        u - the solution to the PDE    
    """
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    #print("lambda=",lmbda)

    
    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step
    u_jm1 = np.zeros(x.size)      # u at previous time step
    
    if method.upper() == 'FE':
        if lmbda > 0.25:
            print('Invalid use of this method')
            return 0
            
        u_j = u_I(x)
        u_jFE = Forward_Euler(lmbda, u_j, u_jp1, mx, mt, a, b, conditions, x)
        if plot == True:
            plotter(x, u_jFE, method)
        return u_jFE
        
    elif method.upper() == 'BE':
        u_j = u_I(x)
        u_jm1BE = Backward_Euler(lmbda, u_jm1, u_j, mx, mt, a, b, conditions, x)
        if plot == True:
            plotter(x, u_jm1BE, method)
        return u_jm1BE
        
    elif method.upper() == 'CN':
        u_j = u_I(x)
        u_jp1CN = Crank_Nicholson(lmbda, u_jp1, u_j, mx, mt, a, b, conditions, x)
        if plot == True:
            plotter(x, u_jp1CN, method)

        return u_jp1CN
    else:
        print('Please enter the initials of an approximation scheme\n FE, BE, CN')
        
    
###############################################################################
###############################################################################

# set numerical parameters
mx = 20     # number of gridpoints in space
mt = 1000   # number of gridpoints in time


# set problem parameters/functions
kappa = 1   # diffusion constant
L=1         # length of spatial domain
T=0.5       # total time to solve for

condition = 'NEUMANN'
condition = 'DIRICHLET'

#Parabolic_PDE(kappa, L, T, u_I, mx, mt, LHBound, RHBound, 'CN', condition, True)
#e, x_val = ErrorVn(23, 'FE', 'dx')

if __name__ == '__main__':
    Parabolic_PDE(kappa, L, T, u_I, mx, mt, LHBound, RHBound, 'CN', condition, True)
