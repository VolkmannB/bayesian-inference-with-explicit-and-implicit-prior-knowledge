import numpy as np
import jax
import jax.numpy as jnp
import functools
import scipy.io

from src.Filtering import reconstruct_trajectory
from src.Vehicle import mu_y, basis_fcn, f_alpha, ctrl_input
from src.Vehicle import time, Vehicle_Algorithm1, Vehicle_Algorithm2
from src.Vehicle import (
    GP_prior_f,
    GP_prior_r,
    key,
    X,
    Y,
    mu_f,
    mu_r,
)


################################################################################
# Online Algorithm
print("\n=== Online Algorithm ===")

key, key_sim = jax.random.split(key)
(
    online_Sigma_X,
    online_Sigma_mu,
    online_GP_stats,
    online_weights,
    _,
    _,
    online_Sigma_Y,
    online_log_likelihood,
) = Vehicle_Algorithm1(key_sim)
online_GP_stats_f = online_GP_stats[0]
online_GP_stats_r = online_GP_stats[1]
del online_GP_stats
online_Sigma_mu_f = online_Sigma_mu[0]
online_Sigma_mu_r = online_Sigma_mu[1]
online_T0_f, online_T1_f, online_T2_f, online_T3_f = online_GP_stats_f
del online_GP_stats_f
online_T0_r, online_T1_r, online_T2_r, online_T3_r = online_GP_stats_r
del online_GP_stats_r
online_Sigma_alpha_f, online_Sigma_alpha_r = jax.vmap(
    jax.vmap(f_alpha, in_axes=(0, None))
)(online_Sigma_X, ctrl_input)

################################################################################
# Offline Algorithm
print("\n=== Offline Algorithm ===")

# reate reference
print("Creating reference trajectory...")
key, key_sim, key_traj = jax.random.split(key, 3)
(
    init_ref_state,
    init_ref_int_var,
    _,
    init_ref_weights,
    init_ref_ancestors,
    _,
    _,
    _,
) = Vehicle_Algorithm1(key_sim)
idx = np.searchsorted(np.cumsum(init_ref_weights), jax.random.uniform(key_traj))
init_ref_state = reconstruct_trajectory(init_ref_state, init_ref_ancestors, idx)
init_ref_int_var = [
    reconstruct_trajectory(init_ref_int_var[i], init_ref_ancestors, idx)
    for i in range(len(init_ref_int_var))
]
init_ref_int_var = tuple(init_ref_int_var)

(
    offline_Sigma_X,
    offline_Sigma_mu,
    offline_weights,
    offline_GP_stats,
    offline_Sigma_Y,
    offline_log_likelihood,
) = Vehicle_Algorithm2(key, init_ref_state, init_ref_int_var)
offline_GP_stats_f = offline_GP_stats[0]
offline_GP_stats_r = offline_GP_stats[1]
del offline_GP_stats
offline_Sigma_mu_f = offline_Sigma_mu[0]
offline_Sigma_mu_r = offline_Sigma_mu[1]
del offline_Sigma_mu
offline_T0_f, offline_T1_f, offline_T2_f, offline_T3_f = offline_GP_stats_f
del offline_GP_stats_f
offline_T0_r, offline_T1_r, offline_T2_r, offline_T3_r = offline_GP_stats_r
del offline_GP_stats_r
offline_Sigma_alpha_f, offline_Sigma_alpha_r = jax.vmap(
    jax.vmap(f_alpha, in_axes=(0, None))
)(offline_Sigma_X, ctrl_input)

################################################################################
# Save Results

# precompute true function for later plotting
alpha_plot = jnp.linspace(-20 / 180 * jnp.pi, 20 / 180 * jnp.pi, 500)
mu_true_plot = jax.vmap(functools.partial(mu_y))(alpha=alpha_plot)
basis_plot = jax.vmap(basis_fcn)(alpha_plot)

# Create save file
mdict = {
    "offline_Sigma_X": offline_Sigma_X,
    "offline_Sigma_Y": offline_Sigma_Y,
    "offline_Sigma_mu_f": offline_Sigma_mu_f,
    "offline_Sigma_mu_r": offline_Sigma_mu_r,
    "offline_Sigma_alpha_f": offline_Sigma_alpha_f,
    "offline_Sigma_alpha_r": offline_Sigma_alpha_r,
    "offline_weights": offline_weights,
    "offline_log_likelihood": offline_log_likelihood,
    "offline_T0_f": offline_T0_f,
    "offline_T1_f": offline_T1_f,
    "offline_T2_f": offline_T2_f,
    "offline_T3_f": offline_T3_f,
    "offline_T0_r": offline_T0_r,
    "offline_T1_r": offline_T1_r,
    "offline_T2_r": offline_T2_r,
    "offline_T3_r": offline_T3_r,
    "online_Sigma_X": online_Sigma_X,
    "online_Sigma_Y": online_Sigma_Y,
    "online_Sigma_mu_f": online_Sigma_mu_f,
    "online_Sigma_mu_r": online_Sigma_mu_r,
    "online_Sigma_alpha_f": online_Sigma_alpha_f,
    "online_Sigma_alpha_r": online_Sigma_alpha_r,
    "online_weights": online_weights,
    "online_log_likelihood": online_log_likelihood,
    "online_T0_f": online_T0_f,
    "online_T1_f": online_T1_f,
    "online_T2_f": online_T2_f,
    "online_T3_f": online_T3_f,
    "online_T0_r": online_T0_r,
    "online_T1_r": online_T1_r,
    "online_T2_r": online_T2_f,
    "online_T3_r": online_T3_r,
    "time": time,
    "alpha_plot": alpha_plot,
    "basis_plot": basis_plot,
    "mu_true_plot": mu_true_plot,
    "prior_T0_f": GP_prior_f[0],
    "prior_T1_f": GP_prior_f[1],
    "prior_T2_f": GP_prior_f[2],
    "prior_T3_f": GP_prior_f[3],
    "prior_T0_r": GP_prior_r[0],
    "prior_T1_r": GP_prior_r[1],
    "prior_T2_r": GP_prior_r[2],
    "prior_T3_r": GP_prior_r[3],
    "X": X,
    "Y": Y,
    "mu_f": mu_f,
    "mu_r": mu_r,
}
scipy.io.savemat("plots/Vehicle.mat", mdict)
