import numpy as np
from matplotlib import pyplot as plt


spacing=0.0000001

def grid(min, max, gridpoints):
    grid = np.linspace(min, max, gridpoints,  dtype=np.complex_)
    return grid

def wavefunc(grid, func):
    func_vec = np.vectorize(func)
    return func_vec(grid)

def potential(grid, func):
    func_vec = np.vectorize(func)
    return func_vec(grid)

def timegrid(max, spacing):
    timegrid = np.arange(0, max, spacing)
    return timegrid

def solver(psi_0, V):
    psi_0_l = np.roll(psi_0,1)
    psi_0_u = np.roll(psi_0,-1)
    psi_0_l[0]=0
    psi_0_u[-1]=0

    b=spacing
    psi_t_x = psi_0 + (-1j*b/h_bar)*(-(h_bar**2/(2*m*a**2))*(psi_0_u-2*psi_0+psi_0_l)+np.multiply(psi_0,V))

    return psi_t_x


max_time = 100
min_grid = 0
max_grid = 100
gridpoints = 50
h_bar=1
m=1
a = 2


x = grid(min_grid, max_grid, gridpoints)
psi_0 = np.ones(gridpoints)

def V(x):
    return 10*(x-50)**2

V = potential(x, V)
timegrid = timegrid(max_time, spacing)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)
psi_0 =  solver(psi_0, V)

plt.plot(x, np.multiply(np.conj(psi_0),psi_0))
#plt.plot(x, V)
plt.show()
