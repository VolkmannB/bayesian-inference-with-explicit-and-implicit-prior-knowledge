import jax
import jax.numpy as jnp
import scipy.io

from src.EMPS import EMPS_PGAS, basis_fcn
from src.EMPS import GP_prior, Y, X
from src.EMPS import EMPS_APF, time



################################################################################
### Offline Algorithm
print("\n=== Offline Algorithm ===")

(
    offline_Sigma_X, 
    offline_Sigma_F, 
    offline_Sigma_Y, 
    offline_weights, 
    offline_GP_stats
    ) = EMPS_PGAS()
offline_T0, offline_T1, offline_T2, offline_T3 = offline_GP_stats
del offline_GP_stats



################################################################################
### Online Algorithm
print("\n=== Online Algorithm ===")

(
    online_Sigma_X, 
    online_Sigma_F, 
    online_Sigma_Y, 
    online_weights, 
    online_GP_stats
    ) = EMPS_APF()
online_T0, online_T1, online_T2, online_T3 = online_GP_stats
del online_GP_stats



################################################################################
# Saving Results

dq_plot = jnp.linspace(-0.15, 0.15, 500)
basis_plot = jax.vmap(basis_fcn)(dq_plot)

# Create save file
mdict = {
    'offline_Sigma_X': offline_Sigma_X,
    'offline_Sigma_F': offline_Sigma_F,
    'offline_Sigma_Y': offline_Sigma_Y,
    'offline_weights': offline_weights,
    'offline_T0': offline_T0,
    'offline_T1': offline_T1,
    'offline_T2': offline_T2,
    'offline_T3': offline_T3,
    'online_Sigma_X': online_Sigma_X,
    'online_Sigma_F': online_Sigma_F,
    'online_Sigma_Y': online_Sigma_Y,
    'online_weights': online_weights,
    'online_T0': online_T0,
    'online_T1': online_T1,
    'online_T2': online_T2,
    'online_T3': online_T3,
    'time': time,
    'dq_plot': dq_plot,
    'basis_plot': basis_plot,
    'prior_T0': GP_prior[0],
    'prior_T1': GP_prior[1],
    'prior_T2': GP_prior[2],
    'prior_T3': GP_prior[3],
    'Y': Y,
    'X': X
}
scipy.io.savemat('plots\EMPS.mat', mdict)