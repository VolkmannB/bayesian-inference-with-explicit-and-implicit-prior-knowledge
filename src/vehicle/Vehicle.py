import numpy as np
import math
import jax
import jax.numpy as jnp
import pytensor.tensor as pt
import pymc.math as pmcm



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
    
    return mu * pt.sin(C * pt.arctan(B*(1-E)*pt.tan(alpha) + E*pt.arctan(B*pt.tan(alpha))))



# side slip
def f_alpha(x, u):
    
    vx_f = pt.sin(u[0])*u[1] + pt.cos(u[0])*x[1]
    vy_f = pt.cos(u[0])*u[1] - pt.sin(u[0])*x[1]
    
    vx_r = u[1]
    vy_r = x[1]
    return pt.arctan2(vy_f, vx_f), pt.arctan2(vy_r, vx_r)


# state space model
def f_dx_sim(x, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r):
    
    F_zf, F_zr = f_Fz(m, g, l_f, l_r)
    alpha_f, alpha_r = f_alpha(x, u)
    mu_yf = mu_y(alpha_f, mu, B_f, C_f, E_f)
    mu_yr = mu_y(alpha_r, mu, B_r, C_r, E_r)
    
    dv_y = 1/m*(F_zf*mu_yf*pt.cos(u[0]) + F_zr*mu_yr + F_zf*mu_x*pt.sin(u[0])) - u[1]*x[0]
    dpsi = 1/I_zz*(l_f*F_zf*mu_yf*pt.cos(u[0]) - l_r*F_zr*mu_yr + l_f*F_zf*mu_x*pt.sin(u[0]))
    
    return pt.stack([dv_y, dpsi])



# time discrete state space pdel with Runge-Kutta-4
def f_x(x, u, dt, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r):
    
    k1 = f_dx_sim(x, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k2 = f_dx_sim(x + dt*k1/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k3 = f_dx_sim(x + dt*k2/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k4 = f_dx_sim(x + dt*k3/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    
    return x + dt/6.0*(k1+2*k2+2*k3+k4)



##### Filter


