import numpy as np
import math
import jax
import jax.numpy as jnp
import functools
from tqdm import tqdm


from src.BayesianInferrence import gaussian_RBF, bump_RBF, generate_Hilbert_BasisFunction


# x = [dpsi, v_y]
# u = [delta, v_x]

##### default arameters
m = 1720.0
I_zz = 1827.5
l_f = 1.16
l_r = 1.47
g = 9.81
mu_x = 0.9
mu = 0.9
B = 10.0
C = 1.9
E = 0.97



##### Simulation

# tire load
def f_Fz(m, l_f, l_r, g):
    
    l = l_f + l_r
    mg = m*g
    F_zf = mg*l_r/l
    F_zr = mg*l_f/l
    
    return F_zf, F_zr



# friction MTF curve
@jax.jit
def mu_y(alpha, mu=mu, B=B, C=C, E=E):
    
    return mu * jnp.sin(C * jnp.arctan(B*(1-E)*jnp.tan(alpha) + E*jnp.arctan(B*jnp.tan(alpha))))



# side slip
@jax.jit
def f_alpha(x, u, l_f=l_f, l_r=l_r):

    vx_f = u[1]
    vy_f = x[1] + x[0]*l_f
    
    vx_r = u[1]
    vy_r = x[1] - x[0]*l_r
    
    return u[0]-jnp.arctan(vy_f/vx_f), -jnp.arctan(vy_r/vx_r)



# state dynamics
def dx(x, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x):
    
    F_zf, F_zr = f_Fz(m, l_f, l_r, g)
    
    dv_y = 1/m * (F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*mu_x*jnp.sin(u[0])) - u[1]*x[0]
    ddpsi = 1/I_zz * (l_f*F_zf*mu_yf*jnp.cos(u[0]) - l_r*F_zr*mu_yr + l_f*F_zf*mu_x*jnp.sin(u[0]))
    
    return jnp.stack([ddpsi, dv_y])



# time discrete state space model with Runge-Kutta-4
@jax.jit
def f_x(x, u, mu_yf, mu_yr, dt, m=m, I_zz=I_zz, l_f=l_f, l_r=l_r, g=g, mu_x=mu_x):
    
    k1 = dx(x, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k2 = dx(x + dt*k1/2.0, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k3 = dx(x + dt*k2/2.0, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k4 = dx(x + dt*k3, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    
    return x + dt/6.0*(k1+2*k2+2*k3+k4)



# measurment model
@jax.jit
def f_y(x, u, mu_yf, mu_yr, m=m, l_f=l_f, l_r=l_r, g=g, mu_x=mu_x, mu=mu, B=B, C=C, E=E):
    
    F_zf, F_zr = f_Fz(m, l_f, l_r, g)
    
    dv_y = 1/m * (F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*mu_x*jnp.sin(u[0])) - u[1]*x[0]
    
    return jnp.array([x[0], dv_y, x[1]])



##### Filtering

# features for front and rear tire
N_basis_fcn = 10
lengthscale = 2 * 20/180*jnp.pi / N_basis_fcn
basis_fcn, spectral_density = generate_Hilbert_BasisFunction(
    N_basis_fcn, 
    np.array([-30/180*jnp.pi, 30/180*jnp.pi]), 
    lengthscale, 
    50
    )

@jax.jit
def features_MTF_front(x, u, l_f=l_f):
    
    vx = u[1]
    vy = x[1] + x[0]*l_f
    alpha = u[0] - jnp.arctan(vy/vx)
    
    return basis_fcn(alpha)

@jax.jit
def features_MTF_rear(x, u, l_r=l_r):
    
    vx = u[1]
    vy = x[1] - x[0]*l_r
    alpha = -jnp.arctan(vy/vx)
    
    return basis_fcn(alpha)