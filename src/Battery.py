import jax
import jax.numpy as jnp
import equinox as eqx

from src.BayesianInferrence import bump_RBF



# parameters
default_para = {
    'R_c': 0.407, # from literature
    'C_c': 43.5, # from literature
    'T_amb': 25, # given
    'V_0': 2.56
}

@jax.jit
class BatterySSM(eqx.Module):
    
    R_c: int
    C_c: int
    T_amb: int
    V_0: int
    
    def __init__(self, R_c, C_c, T_amb, V_0):
        
        self.R_c = R_c
        self.C_c = C_c
        self.T_amb = T_amb
        self.V_0 = V_0
    
    def dx(self, x, I, Q, alpha, beta, R_0):
        
        dz = I / Q
        dV = -alpha*x[1] + beta*I
        dT = (-(x[2] - self.T_amb)/self.R_c + x[1]*I + R_0*I**2)/self.C_c
    
        return jnp.array([dz, dV, dT])
    
    def __call__(self, x, I, Q, alpha, beta, R_0, dt):
        
        k1 = self.dx(x, I, Q, alpha, beta, R_0)
        k2 = self.dx(x+dt*k1/2, I, Q, alpha, beta, R_0)
        k3 = self.dx(x+dt*k2/2, I, Q, alpha, beta, R_0)
        k4 = self.dx(x+dt*k3, I, Q, alpha, beta, R_0)
        
        return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
    def fy(self, x, I, R_0):
        return jnp.hstack([self.V_0 + x[1] + R_0*I, x[3]])
    
model = BatterySSM(**default_para)



# basis function for Voltage model for alpha and beta dependent on SoC
z_ip = jnp.linspace(0, 1, 5)
l_z = z_ip[1] - z_ip[0]

@jax.jit
def features_V(x):
    return bump_RBF(x[0], z_ip, l_z)



# basis functions for resistor model dependent on SoC and current
I_ip = jnp.linspace(-5, 5, 5)
l_I = I_ip[1] - I_ip[0]
zI_ip = jnp.dstack(jnp.meshgrid(z_ip, I_ip, indexing='xy'))
zI_ip = zI_ip.reshape(zI_ip.shape[0]*zI_ip.shape[1], 2)

@jax.jit
def features_R(x,I):
    return bump_RBF(jnp.atleast_2d(jnp.array([x[0],I])), zI_ip, jnp.array([l_z, l_I]))