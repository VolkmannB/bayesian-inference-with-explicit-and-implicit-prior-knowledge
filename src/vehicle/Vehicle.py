import numpy as np
import math
import jax
import jax.numpy as jnp
import functools
from tqdm import tqdm


from src.RGP import gaussian_RBF, bump_RBF


# x = [dpsi, v_y]
# u = [delta, v_x]

##### default arameters
default_para = {
    'dt': 0.01,
    'm': 1720,
    'I_zz': 1827.5431059723351,
    'l_f': 1.1625348837209302,
    'l_r': 1.4684651162790696,
    'g':9.81,
    'mu_x': 0.9,
    'mu': 0.9,
    'B_f': 10.0,
    'C_f': 1.9,
    'E_f': 0.97,
    'B_r': 10.0,
    'C_r': 1.9,
    'E_r': 0.97
}



##### Simulation

# tire load
def f_Fz(**para):
    
    l = para['l_f'] + para['l_r']
    mg = para['m']*para['g']
    F_zf = mg*para['l_r']/l
    F_zr = mg*para['l_f']/l
    
    return F_zf, F_zr



# friction MTF curve
@jax.jit
def mu_y(alpha, mu, B, C, E):
    
    return mu * jnp.sin(C * jnp.arctan(B*(1-E)*jnp.tan(alpha) + E*jnp.arctan(B*jnp.tan(alpha))))



# side slip
@jax.jit
def f_alpha(x, u, **para):

    vx_f = u[1]
    vy_f = x[1] + x[0]*para['l_f']
    
    vx_r = u[1]
    vy_r = x[1] - x[0]*para['l_r']
    
    return u[0]-jnp.arctan(vy_f/vx_f), -jnp.arctan(vy_r/vx_r)



def dx(x, u, mu_yf, mu_yr, **para):
    
    F_zf, F_zr = f_Fz(**para)
    
    dv_y = 1/para['m']*(F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*para['mu_x']*jnp.sin(u[0])) - u[1]*x[0]
    ddpsi = 1/para['I_zz']*(para['l_f']*F_zf*mu_yf*jnp.cos(u[0]) - para['l_r']*F_zr*mu_yr + para['l_f']*F_zf*para['mu_x']*jnp.sin(u[0]))
    
    return jnp.stack([ddpsi, dv_y])



# state space model
def f_dx_sim(x, u, **para):
    
    alpha_f, alpha_r = f_alpha(x, u, **para)
    mu_yf = mu_y(alpha_f, para['mu'], para['B_f'], para['C_f'], para['E_f'])
    mu_yr = mu_y(alpha_r, para['mu'], para['B_r'], para['C_r'], para['E_r'])
    
    return dx(x, u, mu_yf, mu_yr, **para)



# time discrete state space pdel with Runge-Kutta-4
@jax.jit
def f_x_sim(x, u, **para):
    
    k1 = f_dx_sim(x, u, **para)
    k2 = f_dx_sim(x + para['dt']*k1/2.0, u, **para)
    k3 = f_dx_sim(x + para['dt']*k2/2.0, u, **para)
    k4 = f_dx_sim(x + para['dt']*k3, u, **para)
    
    return x + para['dt']/6.0*(k1+2*k2+2*k3+k4)



# measurment model
@jax.jit
def f_y(x, u, **para):
    
    F_zf, F_zr = f_Fz(**para)
    alpha_f, alpha_r = f_alpha(x, u, **para)
    mu_yf = mu_y(alpha_f, para['mu'], para['B_f'], para['C_f'], para['E_f'])
    mu_yr = mu_y(alpha_r, para['mu'], para['B_r'], para['C_r'], para['E_r'])
    
    dv_y = 1/para['m']*(F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*para['mu_x']*jnp.sin(u[0])) - u[1]*x[0]
    
    return jnp.array([x[0], dv_y])



##### Filtering

# features for front and rear tire
vehicle_RBF_ip = jnp.atleast_2d(jnp.linspace(-20/180*jnp.pi, 20/180*jnp.pi, 8)).T
vehicle_lengthscale = vehicle_RBF_ip[1] - vehicle_RBF_ip[0]
H_vehicle = lambda alpha: bump_RBF(alpha, vehicle_RBF_ip, vehicle_lengthscale)

@jax.jit
def features_MTF_front(x, u, **para):
    
    vx = u[1]
    vy = x[1] + x[0]*para['l_f']
    alpha = u[0] - jnp.arctan(vy/vx)
    
    return H_vehicle(alpha)

@jax.jit
def features_MTF_rear(x, u, **para):
    
    vx = u[1]
    vy = x[1] - x[0]*para['l_r']
    alpha = -jnp.arctan(vy/vx)
    
    return H_vehicle(alpha)



# time discrete SSM with Runge-Kutta 4
@jax.jit
def fx_filter(x, u, mu_yf, mu_yr, **para):
    
    k1 = dx(x, u, mu_yf, mu_yr, **para)
    k2 = dx(x + para['dt']*k1/2.0, u, mu_yf, mu_yr, **para)
    k3 = dx(x + para['dt']*k2/2.0, u, mu_yf, mu_yr, **para)
    k4 = dx(x + para['dt']*k3, u, mu_yf, mu_yr, **para)
    
    return x + para['dt']/6.0*(k1 + 2*k2 + 2*k3 + k4)


@jax.jit
def fy_filter(x, u, mu_yf, mu_yr, **para):
    
    F_zf, F_zr = f_Fz(**para)
    
    dv_y = 1/para['m']*(F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*para['mu_x']*jnp.sin(u[0])) - u[1]*x[0]
    
    return jnp.array([x[0], dv_y, x[1]])