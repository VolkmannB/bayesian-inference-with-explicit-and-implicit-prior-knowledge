import numpy as np
import jax.numpy as jnp
import jax

from src.RGP import gaussian_RBF


#### This section defines the simulated single mass oscillator

def F_spring(x, **para):
    return para['c1'] * x + para['c2'] * x**3



def F_damper(dx, **para):
    return para['d1']*dx * (1/(1+para['d2']*dx*jnp.tanh(dx)))



def dx(x, F, F_sd, **para):
    return jnp.array(
        [x[1], -F_sd/para['m'] + F/para['m'] + 9.81]
    )



def dx_sim(x, F, **para):
    
    F_s = F_spring(x[0], **para)
    F_d= F_damper(x[1], **para)
    
    return dx(x, F, F_s+F_d, **para)



@jax.jit
def f_x_sim(x, F, **para):
    
    # Runge-Kutta 4
    k1 = dx_sim(x, F, **para)
    k2 = dx_sim(x+para['dt']/2.0*k1, F, **para)
    k3 = dx_sim(x+para['dt']/2.0*k2, F, **para)
    k4 = dx_sim(x+para['dt']*k3, F, **para)
    x = x + para['dt']/6.0*(k1+2*k2+2*k3+k4) 
    
    return x
    
    
    
#### this section defines functions related to the state space model of the filtering problem

# RBF for GP (the feature vector)
x_points = np.arange(-5., 5.1, 1.)
dx_points = np.arange(-5., 5.1, 1.)
ip = np.dstack(np.meshgrid(x_points, dx_points, indexing='xy'))
ip = ip.reshape(ip.shape[0]*ip.shape[1], 2)
H = lambda x: gaussian_RBF(
    x,
    inducing_points=jnp.asarray(ip),
    lengthscale=jnp.array([1.])
)



# the ode for the KF
def dx_KF(x, F, theta, **para):
    
    x_dot = dx(x[:2], F, x[2], **para)
    
    dH_dt = jax.jvp(H, (x[:2],), (x_dot,))[1]
    
    return jnp.hstack([x_dot, dH_dt@theta])
    
    

# time discrete state space model for the filter with Runge-Kutta 4 integration
@jax.jit
def fx_KF(x, F, theta, **para):
    
    k1 = dx_KF(x, F, theta, **para)
    k2 = dx_KF(x+para['dt']*k1/2, F, theta, **para)
    k3 = dx_KF(x+para['dt']*k2/2, F, theta, **para)
    k4 = dx_KF(x+para['dt']*k3, F, theta, **para)
    
    return x + para['dt']/6*(k1 + 2*k2 + 2*k3 + k4)