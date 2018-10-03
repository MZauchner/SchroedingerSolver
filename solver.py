import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import signal


spacing=0.001
def grid(min, max, gridpoints):
    grid = np.linspace(min, max, gridpoints,  dtype=np.complex_)
    return grid

def wavefunc(grid, func, *args):
    func_vec = np.vectorize(func)
    return func_vec(grid, *args)+0j

def potential(grid, func):
    func_vec = np.vectorize(func)
    return func_vec(grid)+0j

def timegrid(max, spacing):
    timegrid = np.arange(0, max, spacing)
    return timegrid

def solver(psi_0, V):
    psi_0_l = np.roll(psi_0,1)
    psi_0_u = np.roll(psi_0,-1)
    psi_0_l[0]=0+0j
    psi_0_u[-1]=0+0j

    b=spacing
    psi_t_x = psi_0 + (-1j*b/h_bar)*(((-h_bar**2/(2*m*a**2))*(psi_0_u-2*psi_0+psi_0_l))+np.multiply(psi_0,V))
    return psi_t_x/np.sqrt((np.sum(np.multiply(np.conj(psi_t_x),psi_t_x))))/2


max_time = 100
min_grid = 0
max_grid = 1000
gridpoints = 500
h_bar=1
m=1
a = 2
def gaussian(x, mu, sig):
    return 1./(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def V(x):
    return (0.001*(x-500))

x = grid(min_grid, max_grid, gridpoints)
psi_0 = signal.gaussian(500, std=10)+0j



C = potential(x, V)

print(C.dtype)
print(psi_0.dtype)
print(x.dtype)
timegrid = timegrid(max_time, spacing)
for i in range(1000000):
    psi_0 =  solver(psi_0, C)






print(psi_0.dtype)
plt.plot(x, np.power(np.absolute(psi_0),2))
#plt.plot(x, C)
plt.show()
