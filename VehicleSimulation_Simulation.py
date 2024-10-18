import jax
import jax.numpy as jnp
import functools
import scipy.io

from src.Vehicle import Vehicle_simulation, mu_y, basis_fcn
from src.Vehicle import time, Vehicle_PGAS
from src.Vehicle import GP_prior_f, GP_prior_r, Vehicle_APF



################################################################################
# Simulation

X, Y, mu_f, mu_r = Vehicle_simulation()



################################################################################
# Offline Algorithm
print("\n=== Offline Algorithm ===")

(
    offline_Sigma_X, 
    offline_Sigma_mu_f, 
    offline_Sigma_mu_r, 
    offline_Sigma_alpha_f, 
    offline_Sigma_alpha_r, 
    offline_weights, 
    offline_GP_stats_f,
    offline_GP_stats_r
    ) = Vehicle_PGAS(Y)
offline_T0_f, offline_T1_f, offline_T2_f, offline_T3_f = offline_GP_stats_f
del offline_GP_stats_f
offline_T0_r, offline_T1_r, offline_T2_r, offline_T3_r = offline_GP_stats_r
del offline_GP_stats_r



################################################################################
# Online Algorithm
print("\n=== Online Algorithm ===")

(
    online_Sigma_X, 
    online_Sigma_mu_f, 
    online_Sigma_mu_r, 
    online_Sigma_alpha_f, 
    online_Sigma_alpha_r, 
    online_weights, 
    online_GP_stats_f,
    online_GP_stats_r
    ) = Vehicle_APF(Y)
online_T0_f, online_T1_f, online_T2_f, online_T3_f = online_GP_stats_f
del online_GP_stats_f
online_T0_r, online_T1_r, online_T2_r, online_T3_r = online_GP_stats_r
del online_GP_stats_r



################################################################################
# Save Results

# precompute true function for later plotting
alpha_plot = jnp.linspace(-20/180*jnp.pi, 20/180*jnp.pi, 500)
mu_true_plot = jax.vmap(functools.partial(mu_y))(alpha=alpha_plot)
basis_plot = jax.vmap(basis_fcn)(alpha_plot)

# Create save file
mdict = {
    'offline_Sigma_X': offline_Sigma_X,
    'offline_Sigma_mu_f': offline_Sigma_mu_f,
    'offline_Sigma_mu_r': offline_Sigma_mu_r,
    'offline_Sigma_alpha_f': offline_Sigma_alpha_f,
    'offline_Sigma_alpha_r': offline_Sigma_alpha_r,
    'offline_weights': offline_weights,
    'offline_T0_f': offline_T0_f,
    'offline_T1_f': offline_T1_f,
    'offline_T2_f': offline_T2_f,
    'offline_T3_f': offline_T3_f,
    'offline_T0_r': offline_T0_r,
    'offline_T1_r': offline_T1_r,
    'offline_T2_r': offline_T2_r,
    'offline_T3_r': offline_T3_r,
    'online_Sigma_X': online_Sigma_X,
    'online_Sigma_mu_f': online_Sigma_mu_f,
    'online_Sigma_mu_r': online_Sigma_mu_r,
    'online_Sigma_alpha_f': online_Sigma_alpha_f,
    'online_Sigma_alpha_r': online_Sigma_alpha_r,
    'online_weights': online_weights,
    'online_T0_f': online_T0_f,
    'online_T1_f': online_T1_f,
    'online_T2_f': online_T2_f,
    'online_T3_f': online_T3_f,
    'online_T0_r': online_T0_r,
    'online_T1_r': online_T1_r,
    'online_T2_r': online_T2_f,
    'online_T3_r': online_T3_r,
    'time': time,
    'alpha_plot': alpha_plot,
    'basis_plot': basis_plot,
    'mu_true_plot': mu_true_plot,
    'prior_T0_f': GP_prior_f[0],
    'prior_T1_f': GP_prior_f[1],
    'prior_T2_f': GP_prior_f[2],
    'prior_T3_f': GP_prior_f[3],
    'prior_T0_r': GP_prior_r[0],
    'prior_T1_r': GP_prior_r[1],
    'prior_T2_r': GP_prior_r[2],
    'prior_T3_r': GP_prior_r[3],
    'X': X,
    'Y': Y,
    'mu_f': mu_f,
    'mu_r': mu_r
}
scipy.io.savemat('plots\Vehicle.mat', mdict)