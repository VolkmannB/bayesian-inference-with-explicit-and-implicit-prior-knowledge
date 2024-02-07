import numpy as np
import math
import jax
import jax.numpy as jnp



# x = [dpsi, v_y]
# u = [delta, v_x]

##### Simulation

# tire load
def f_Fz(m, g, l_f, l_r):
    
    l = l_f + l_r
    F_zf = m*g*l_r/l
    F_zr = m*g*l_f/l
    return F_zf, F_zr



# friction MTF curve
def mu_y(alpha, mu, B, C, E):
    
    return mu * jnp.sin(C * jnp.arctan(B*(1-E)*jnp.tan(alpha) + E*jnp.arctan(B*jnp.tan(alpha))))



# side slip
def f_alpha(x, u):

    vx_f = u[1]
    vy_f = x[1]
    
    vx_r = u[1]
    vy_r = x[1]
    
    return u[0]-jnp.arctan2(vx_f, vy_f), jnp.arctan2(vx_r, vy_r)


# state space model
def f_dx_sim(x, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r):
    
    F_zf, F_zr = f_Fz(m, g, l_f, l_r)
    alpha_f, alpha_r = f_alpha(x, u)
    mu_yf = mu_y(alpha_f, mu, B_f, C_f, E_f)
    mu_yr = mu_y(alpha_r, mu, B_r, C_r, E_r)
    
    dv_y = 1/m*(F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*mu_x*jnp.sin(u[0])) - u[1]*x[0]
    dpsi = 1/I_zz*(l_f*F_zf*mu_yf*jnp.cos(u[0]) - l_r*F_zr*mu_yr + l_f*F_zf*mu_x*jnp.sin(u[0]))
    
    return jnp.stack([dv_y, dpsi])



# time discrete state space pdel with Runge-Kutta-4
@jax.jit
def f_x(x, u, dt, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r):
    
    k1 = f_dx_sim(x, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k2 = f_dx_sim(x + dt*k1/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k3 = f_dx_sim(x + dt*k2/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k4 = f_dx_sim(x + dt*k3/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    
    return x + dt/6.0*(k1+2*k2+2*k3+k4)



##### Parameters (dt, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
para = jnp.array([
    0.01, # dt
    1720, # m
    1827.5431059723351, # J_z
    1.1625348837209302, # l_f
    1.4684651162790696, #l_r
    9.81, # g
    0.9, # mu_x
    0.9, # mu
    10.0, # B_f
    1.9, # C_f
    0.97, # E_f
    10.0, # B_r
    1.9, # C_r
    0.97, # E_r
])



##### Filter


