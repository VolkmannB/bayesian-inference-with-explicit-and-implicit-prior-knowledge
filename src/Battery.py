import jax
import jax.numpy as jnp
import equinox as eqx

from src.BayesianInferrence import bump_RBF



# parameters
default_para = dict(
    T_amb = 27, # given
    V_0 = 2.4125, # interpolated from product specification V_nom at 1.0C and V_cutoff at 0.2C
    R_c = 14.0, # from literature
    C_c = 43.5, # from literature
    Q_cap = 3500e-3 * 3.6, # capacity in As (guess)
)
para_train = dict(
    Q_cap = 3450e-3 * 60 * 60, # capacity in As (guess)
    R_0 = 30e-3, # serial resistor in Ohm (guess)
    C_1 = 1.5e3, # cell in F (guess)
    R_1 = 0.022 # cell resistor in Ohm (guess)
)



class BatterySSM(eqx.Module):
    
    T_amb: float
    V_0: float
    R_c: float
    C_c: float
    Q_cap: float
    
    def __call__(self, x, I, R_1, C_1, R_0, dt):
        
        x_new = fx(
            x=x,
            I=I,
            R_1=R_1,
            C_1=C_1,
            R_0=R_0,
            T_amb=self.T_amb,
            R_c=self.R_c,
            C_c=self.C_c,
            dt=dt
        )
        
        return x_new
    
    @jax.jit
    def fy(self, x, I, R_0):
        return jnp.hstack([self.V_0 + x[0] + R_0*I, x[3]])



def dx(x, I, R_1, C_1, R_0, T_amb, R_c, C_c):
        
        dz = I / C_1 - x[0] / R_1 / C_1
        dT = (-(x[2] - T_amb)/R_c + x[0]*I + R_0*I**2)/C_c
    
        return jnp.array([dz, dT])



@jax.jit
def fx(x, I, R_1, C_1, R_0, T_amb, R_c, C_c, dt):
        
        k1 = dx(x, I, R_1, C_1, R_0, T_amb, R_c, C_c)
        k2 = dx(x+dt*k1/2, I, R_1, C_1, R_0, T_amb, R_c, C_c)
        k3 = dx(x+dt*k2/2, I, R_1, C_1, R_0, T_amb, R_c, C_c)
        k4 = dx(x+dt*k3, I, R_1, C_1, R_0, T_amb, R_c, C_c)
        
        return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
def fy(x, I, R_0, V_0):
    return jnp.hstack([V_0 + x[1] + R_0*I, x[2]])



# basis function for Voltage model for alpha and beta dependent on SoC
z_ip = jnp.linspace(0, 1.2, 11)
l_z = z_ip[1] - z_ip[0]

@jax.jit
def basis_fcn(x):
    return bump_RBF(x[0], z_ip, l_z)



# # basis functions for resistor model dependent on SoC and current
# I_ip = jnp.linspace(-5, 5, 5)
# l_I = I_ip[1] - I_ip[0]
# zI_ip = jnp.dstack(jnp.meshgrid(z_ip, I_ip, indexing='xy'))
# zI_ip = zI_ip.reshape(zI_ip.shape[0]*zI_ip.shape[1], 2)

# @jax.jit
# def features_R(x,I):
#     return C2_InversePoly_RBF(jnp.atleast_2d(jnp.array([x[0],I])), zI_ip, jnp.array([l_z, l_I]))