# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import scipy.linalg as linalg

global anim

def calc_NumSol_AdvDiff_1D(Nx, Nt, T, u, D, L, M):
    '''
    Calculates a numerical solution to the advection diffusion equation in 1D vertically. 
    
    Parameters
    ----------
    Nx : int
        The number of steps along the x axis.
    Nt : int
        The number of time steps.
    T : float
        The total time iterated over.
    u : float
        The advection velocity (would be some constant related to gravity in this case).
    D : float
        Diffusion coefficient.
    L : Float
        The length of the 'rod' or domain we are looking at.
    M : float
        Initial Mass of the gas..

    Returns
    -------
    C : Array of size Nt x Nx of integers
        a Matrix of the 2D solution. Rows represent time steps, columns represent x values

    '''
    C = np.zeros((Nt, Nx))
    C[0, Nx//2] = M/(L/Nx)
    C[0, Nx//2 + 1] = M/(L/Nx)
    
    dx = L / (Nx - 1)
    dt = T / Nt
    
    dC_dx = np.zeros(Nx)
    d2C_dx2 = np.zeros(Nx)
    
    for t in range(1, Nt-1):
        for x in range(1, Nx-1):
            dC_dx[x] = (C[t-2, x+1] - C[t-1, x-1]) / (2 * dx)
            d2C_dx2[x] = (C[t-1, x+1] - 2 * C[t-1, x] + C[t-1, x-1]) / dx**2
        C[t] = C[t-1] - u * dC_dx * dt + D * d2C_dx2 * dt 
    
    return C

def calc_MaxSensorDists_1D(C, L, Nx, T, Nt, M):
    '''
    Calculates the distances sensors can be placed (assuming centred around 0 might need to change).
    
    Parameters
    ----------
    C : Array of size Nt x Nx of integers
        a Matrix of the 2D solution. Rows represent time steps, columns represent x values
    L : Float
        The length of the 'rod' or domain we are looking at.
    Nx : int
        the number of steps along the x axis.
    T : float
        The total time iterated over.
    Nt : int
        The number of time steps.
    M : float
        Initial Mass of the gas.

    Returns
    -------
    list
        Two values, the lowest most and upper most limit where a sensor can be placed and
        the contamination is > 1/100 of the initial mass

    '''
    xposes = np.zeros((Nt, 2))
    for t in range(0, Nt-1):
        if max(C[t]) < M/100:
            xposes[t] = [0, 0]
        else: 
            occurance = 1
            for x in range(0, Nx - 2):
                if (C[t, x] < M/100 and C[t, x+1] > M/100) or (C[t, x] > M/100 and C[t, x+1] < M/100):
                    
                    if occurance == 1:
                        xposes[t, 0] = ((L/Nx) * x)-L/2
                        occurance = 2
                    else:
                        xposes[t, 1] = ((L/Nx) * x)-L/2
                        
        occurance = 1
    return [min(xposes[:,0]), max(xposes[:, 1])]



def calc_NumSol_AdvDiff_2D(Nx, Ny, Nt, Lx, Ly, T, D, g):
    dx = Lx / Nx
    dy = Ly / Ny
    dt = T / Nt

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    # Initial condition: concentration blob in the center
    C = np.zeros((Ny, Nx))
    C[Ny // 2 - 5:Ny // 2 + 5, Nx // 2 - 5:Nx // 2 + 5] = 1

    for t in range(Nt):
        # Compute spatial derivatives
        dC_dx = (np.roll(C, shift=-1, axis=1) - np.roll(C, shift=1, axis=1)) / (2 * dx)
        dC_dy = (np.roll(C, shift=-1, axis=0) - np.roll(C, shift=1, axis=0)) / (2 * dy)

        d2C_dx2 = (np.roll(C, shift=-1, axis=1) - 2 * C + np.roll(C, shift=1, axis=1)) / dx ** 2
        d2C_dy2 = (np.roll(C, shift=-1, axis=0) - 2 * C + np.roll(C, shift=1, axis=0)) / dy ** 2

        # Advection-diffusion update
        C += dt * (D * (d2C_dx2 + d2C_dy2) - g * dC_dy)

    return C



def AnalyticSol_Equation_2D(x, t, D, M):
    '''
    The Analytical solution to the difusion equation given a starting solution 
    of M at x=0, t=0 and 0 for all other x

    Parameters
    ----------
    x : float/array of floats
        a given x value.
    t : t
        a given time value.
    D : float
        Diffusion coefficient
    M : float
        mass of gass.

    Returns
    -------
    float or array of floats
        returns the analytical solutions of the equation given inputs
        
        

    '''
    return M*((D/(4*np.pi*t))**(3/2))*(np.e**((-D*x**2)/(4*t)))


def AnalyticSol_Equation_1D(x, t, D, M):
    
    return (M/((4*np.pi*D*t)**(1/2)))*np.exp(-(x**2)/(4*t*D))
    

def calc_AnalSol_Diff_1D(Nx, Nt, T, D, L, M):
    C = np.zeros((Nt, Nx))
    C[0, Nx//2] = M#/(L/Nx)
    #C[0, Nx//2 + 1] = M/(L/Nx)
    
    x_vals = np.linspace(-L/2, L/2, Nx)
    t_vals = np.linspace(0, T, Nt)
    
    for t in range(1, Nt-1):
            C[t] = AnalyticSol_Equation_1D(x_vals, t_vals[t], D, M)
    
    return C


def calc_NumSol_Diff_D1_FEuler(Nx, Nt, T, D, L, M):
    C = np.zeros((Nt, Nx))
    C[0, Nx//2] = M/(L/Nx)
    C[0, Nx//2 + 1] = M/(L/Nx)
    
    dx = L / (Nx)
    dt = T / Nt
    
    for t in range(0, Nt - 2):
        for x in range(1, Nx - 1):
            C[t+1, x] = C[t, x] + D * (dt / dx**2) * (C[t, x - 1] - 2 * C[t, x] + C[t, x + 1])
    
    return C

def calc_NumSol_Diff_D1_BEuler(Nx, Nt, T, D, L, M):
    x = np.linspace(-L/2, L/2, Nx)
    t = np.linspace(0, T, Nt)
    
    C = np.zeros((Nt, Nx))
    C[0, Nx//2] = M/(L/Nx)
    C[0, Nx//2 + 1] = M/(L/Nx)
    
    A = np.zeros((Nx, Nx))
    b = np.zeros(Nx)
    
    dx = L / (Nx)
    dt = T / Nt
    
    C = dx/dt**2
    
    for i in range(1, Nx-1):
        A[i, i-1] = -C
        A[i, i+1] = -C
        A[i, i] = 1+2*C
        
    A[0, 0] = 0
    A[Nx-1, Nx-1] = 0
    
    #Ainv = np.linalg.inv(A)
    
    for t in range(0, Nt-1):
        for i in range(1, Nx-1):
            b[i] = -C[t, i]
        b[0] = b[Nx - 1] = 0
        u[:] = linalg.solve(A, b)
        
        
        C[t, :] = u
    
    return C

def Animate_1Line(C, L):
    '''
    Animates one line, the solution to C over time

    Parameters
    ----------
    C : Array of size Nt x Nx of integers
        a Matrix of the 2D solution. Rows represent time steps, columns represent x values
    L : Float
        The length of the 'rod' or domain we are looking at.

    Returns
    -------
    line : matplotlib line
        Line to be plotted.

    '''
    global anim
    Nx = len(C[0])
    fig, ax = plt.subplots()
    line, = ax.plot(np.linspace(-L/2, L/2, Nx), np.zeros(Nx)) #NEED TO FIX
    
    ax.set_xlabel('x')
    ax.set_ylabel('concentration')
    ax.set_title('1D_Diffusion')
    ax.set_ylim(0, 10)
    
    
    def update(i):
            
        line.set_ydata(C[2*i])
        return line,
        
    anim = FuncAnimation(fig, update, frames=range(len(C[:, 0])), blit=True)

    plt.show()
    
    
def Animate_2Line(C1, C2, L, line1_title, line2_title):
    global anim
    Nx = len(C1[0])
    fig, ax = plt.subplots()
    line1, = ax.plot(np.linspace(-L/2, L/2, Nx), np.zeros(Nx), label = line1_title) #NEED TO FIX
    line2, = ax.plot(np.linspace(-L/2, L/2, Nx), np.zeros(Nx), label = line2_title)
    ax.set_xlabel('x')
    ax.set_ylabel('concentration')
    ax.set_title('1D_Diffusion')
    ax.legend()
    ax.set_ylim(-0.5, 7)
    
    def update(i):
            
        line1.set_ydata(C1[i])
        line2.set_ydata(C2[i])
        return line1, line2,
        
    anim = FuncAnimation(fig, update, frames=range(len(C1[:, 0])), blit=True)

    
    anim.save('numvanal2.gif', fps=10, extra_args=['-vcodec', 'libx264'])
    
    plt.show()
    
def draw_one_line(C1, C2, t, L):
    Nx = len(C1[0])
    fig, ax = plt.subplots()
    line1, = ax.plot(np.linspace(-L/2, L/2, Nx), C1[t]) #NEED TO FIX
    line2, = ax.plot(np.linspace(-L/2, L/2, Nx), C2[t])
    ax.set_ylim(-0.5, 7)
    
    plt.show()



# Parameters of 2D simulation
Nx, Ny, Nt = 100, 100, 100
Lx, Ly, T = 10.0, 10.0, 10.0
D = 0.1
g = 9.81

# Run the simulation
C_final = calc_NumSol_AdvDiff_2D(Nx, Ny, Nt, Lx, Ly, T, D, g)





L = 10.0  # Length of the spatial domain
T = 10 # Total time
Nx = 200  # Number of points
Nt = 2000 # Number of time steps
u = 1.0   # Advection velocity
D = 0.1   # Diffusion coefficient
M = 1
 
print(np.linspace(0, T, Nt))

# Discretization
dx = L / (Nx - 1)
dt = T / Nt


Sol1 = calc_AnalSol_Diff_1D(Nx, Nt, T, D, L, M)
#Sol2 = calc_NumSol_AdvDiff_1D(Nx, Nt, T, u, D, L, M)
Sol3 = calc_NumSol_Diff_D1_FEuler(Nx, Nt, T, D, L, M)
#Sol4 = calc_NumSol_Diff_D1_BEuler(Nx, Nt, T, D, L, M)

#Animate_1Line(Sol4, L)
Animate_2Line(Sol1, Sol3, L, "Analytic", "Numerical")
#draw_one_line(Sol1, Sol3, 0, L)
#draw_one_line(Sol1, Sol3, 1, L)
#print(calc_MaxSensorDists_1D(Sol1, L, Nx, T, Nt, M))
#print(calc_MaxSensorDists_1D(Sol2, L, Nx, T, Nt, M))

