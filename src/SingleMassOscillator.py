import numpy as np
import jax.numpy as jnp
import jax

from src.BayesianInferrence import gaussian_RBF, bump_RBF, generate_Hilbert_BasisFunction


#### This section defines the simulated single mass oscillator


m=2.0
c1=10.0
c2=2.0
d1=0.7
d2=0.4
C = np.array([[1,0],[0,0]])


def F_spring(x):
    return c1 * x + c2 * x**3



def F_damper(dx):
    return d1*dx * (1/(1+d2*dx*jnp.tanh(dx)))



def dx(x, F, F_sd, p):
    return jnp.array(
        [x[1], -F_sd/p + F/p + 9.81]
    )



@jax.jit
def f_x(x, F, F_sd, dt, p=m):
    
    # Runge-Kutta 4
    k1 = dx(x, F, F_sd, p)
    k2 = dx(x+dt/2.0*k1, F, F_sd, p)
    k3 = dx(x+dt/2.0*k2, F, F_sd, p)
    k4 = dx(x+dt*k3, F, F_sd, p)
    x = x + dt/6.0*(k1+2*k2+2*k3+k4) 
    
    return x
    
@jax.jit
def f_y(x):
    return C @ x
    
    
#### this section defines functions related to the state space model of the filtering problem

# RBF for GP (the feature vector)
N_ip = 5
x_points = np.linspace(-7., 7., N_ip)
dx_points = np.linspace(-7., 7., N_ip)
ip = np.dstack(np.meshgrid(x_points, dx_points, indexing='xy'))
ip = ip.reshape(ip.shape[0]*ip.shape[1], 2)
H = lambda x: bump_RBF(
    jnp.atleast_2d(x),
    loc=jnp.asarray(ip),
    lengthscale=x_points[1]-x_points[0]
)
N_ip = ip.shape[0]

# N_ip = 41
# H, sd = generate_Hilbert_BasisFunction(N_ip, np.array([[-5, 5],[-5, 5]]), 1, 10**2)