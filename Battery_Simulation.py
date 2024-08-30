import jax
import jax.numpy as jnp
import scipy.io

from src.Battery import Battery_PGAS, basis_fcn, scale_C1, scale_R1
from src.Battery import GP_prior, offset_C1, offset_R1, Y
from src.Battery import Battery_APF, time



################################################################################
### Offline Algorithm

(
    offline_Sigma_X, 
    offline_Sigma_C1R1, 
    offline_Sigma_Y, 
    offline_weights, 
    offline_GP_stats
    ) = Battery_PGAS()
offline_T0, offline_T1, offline_T2, offline_T3 = offline_GP_stats
del offline_GP_stats



################################################################################
### Online Algorithm

(
    online_Sigma_X, 
    online_Sigma_C1R1, 
    online_Sigma_Y, 
    online_weights, 
    online_GP_stats
    ) = Battery_APF()
online_T0, online_T1, online_T2, online_T3 = online_GP_stats
del online_GP_stats



################################################################################
# Saving Results

X_plot = jnp.linspace(0, 2.2, 500)
basis_plot = jax.vmap(basis_fcn)(X_plot)

# Create save file
mdict = {
    'offline_Sigma_X': offline_Sigma_X,
    'offline_Sigma_C1R1': offline_Sigma_C1R1,
    'offline_Sigma_Y': offline_Sigma_Y,
    'offline_weights': offline_weights,
    'offline_T0': offline_T0,
    'offline_T1': offline_T1,
    'offline_T2': offline_T2,
    'offline_T3': offline_T3,
    'online_Sigma_X': online_Sigma_X,
    'online_Sigma_C1R1': online_Sigma_C1R1,
    'online_Sigma_Y': online_Sigma_Y,
    'online_weights': online_weights,
    'online_T0': online_T0,
    'online_T1': online_T1,
    'online_T2': online_T2,
    'online_T3': online_T3,
    'time': time,
    'X_plot': X_plot,
    'basis_plot': basis_plot,
    'prior_T0': GP_prior[0],
    'prior_T1': GP_prior[1],
    'prior_T2': GP_prior[2],
    'prior_T3': GP_prior[3],
    'scale_C1': scale_C1,
    'scale_R1': scale_R1,
    'offset_C1': offset_C1,
    'offset_R1': offset_R1,
    'Y': Y
}
scipy.io.savemat('plots\Battery.mat', mdict)