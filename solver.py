import numpy as np
from matplotlib import pyplot as plt



def grid(min, max, gridpoints):
    grid = np.linspace(min, max, gridpoints)
    return grid

def wavefunc(grid, func):
    func_vec = np.vectorize(myfunc)
    return func_vec(grid)

def potential(grid, func):
    func_vec = np.vectorize(myfunc)
    return func_vec(grid)

def timegrid(max, spacing):
    timegrid = np.arange(0, max, spacing)
    return timegrid

def solver(psi_0, V):
    h_bar=1
    m=1
    spacing=0.001
    max_time = 100
    min_grid = 0
    max_grid = 100
    gridpoints = 10000
    a = (max_grid-min_grid)/gridpoints

    x = grid(min_grid, max_grid, gridpoints)
    timegrid = timegrid(max_time, spacing)
    psi_0_l = np.roll(psi_0,1)
    psi_0_u = np.roll(psi_0,-1)
    psi_0_l[0]=0
    psi_0_l[-1]=0
    b = spacing
    psi_t_x = psi_0 - (1j*b/h_bar)*(-(h_bar**2/(2*m*a**2))*(psi_0_u-2*psi_0+psi_0_l)+np.multiply(psi_0,V))

    return psi_t_x
