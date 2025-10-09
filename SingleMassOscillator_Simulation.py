import numpy as np
import jax
import scipy.io

from src.Filtering import reconstruct_trajectory
from src.SingleMassOscillator import F_spring, F_damper, basis_fcn, time, key
from src.SingleMassOscillator import (
    GP_prior,
    SMO_Algorithm1,
    X,
    Y,
    F_sd,
    SMO_Algorithm2,
)


################################################################################
### Online Algorithm
print("\n=== Online Algorithm ===")

# Run auxiliary Particle filter
key, key_sim = jax.random.split(key)
(
    online_Sigma_X,
    online_Sigma_F,
    online_GP_stats,
    online_weights,
    _,
    _,
    online_Sigma_Y,
    online_log_likelihood,
) = SMO_Algorithm1(key_sim)
online_Sigma_F = online_Sigma_F[0]
online_T0, online_T1, online_T2, online_T3 = online_GP_stats[0]
del online_GP_stats


################################################################################
### Offline Algorithm
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
) = SMO_Algorithm1(key_sim)
idx = np.searchsorted(np.cumsum(init_ref_weights), jax.random.uniform(key_traj))
init_ref_state = reconstruct_trajectory(init_ref_state, init_ref_ancestors, idx)
init_ref_int_var = [
    reconstruct_trajectory(init_ref_int_var[i], init_ref_ancestors, idx)
    for i in range(len(init_ref_int_var))
]
init_ref_int_var = tuple(init_ref_int_var)

# Run PGAS
(
    offline_Sigma_X,
    offline_Sigma_F,
    offline_weights,
    offline_GP_stats,
    offline_Sigma_Y,
    offline_log_likelihood,
) = SMO_Algorithm2(key, init_ref_state, init_ref_int_var)
offline_Sigma_F = offline_Sigma_F[0]
offline_T0, offline_T1, offline_T2, offline_T3 = offline_GP_stats[0]
del offline_GP_stats


################################################################################
# Save Results

# precompute input space to function for later plotting
x_plt = np.linspace(-3.5, 3.5, 50)
dx_plt = np.linspace(-3.5, 3.5, 50)
grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing="xy")

X_plot = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
basis_plot = jax.vmap(basis_fcn)(X_plot)

# true spring damper force
F_sd_true_plot = jax.vmap(F_spring)(X_plot[:, 0]) + jax.vmap(F_damper)(
    X_plot[:, 1]
)

# Create save file
mdict = {
    "offline_Sigma_X": offline_Sigma_X,
    "offline_Sigma_Y": offline_Sigma_Y,
    "offline_Sigma_F": offline_Sigma_F,
    "offline_weights": offline_weights,
    "offline_log_likelihood": offline_log_likelihood,
    "offline_T0": offline_T0,
    "offline_T1": offline_T1,
    "offline_T2": offline_T2,
    "offline_T3": offline_T3,
    "online_Sigma_X": online_Sigma_X,
    "online_Sigma_Y": online_Sigma_Y,
    "online_Sigma_F": online_Sigma_F,
    "online_weights": online_weights,
    "online_log_likelihood": online_log_likelihood,
    "online_T0": online_T0,
    "online_T1": online_T1,
    "online_T2": online_T2,
    "online_T3": online_T3,
    "time": time,
    "X_plot": X_plot,
    "basis_plot": basis_plot,
    "F_sd_true_plot": F_sd_true_plot,
    "prior_T0": GP_prior[0],
    "prior_T1": GP_prior[1],
    "prior_T2": GP_prior[2],
    "prior_T3": GP_prior[3],
    "X": X,
    "Y": Y,
    "F_sd": F_sd,
}
scipy.io.savemat("plots/SingleMassOscillator.mat", mdict)
