import jax
import jax.numpy as jnp
import numpy as np
import scipy.io

from src.Filtering import reconstruct_trajectory
from src.EMPS import (
    GP_prior,
    Y,
    X,
    EMPS_Algorithm1,
    EMPS_Algorithm2,
    time,
    basis_fcn,
    key,
    EMPS_PGAS_baseline,
    basis_fcn_f_PGAS,
    GP_prior_PGAS,
    ctrl_input,
    EMPS_Validation_Simulation,
)
import src.BayesianInferrence as BI


################################################################################
### Online Algorithm
print("\n=== Online Algorithm ===")

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
) = EMPS_Algorithm1(key_sim)
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
) = EMPS_Algorithm1(key_sim)
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
) = EMPS_Algorithm2(key, init_ref_state, init_ref_int_var)
offline_Sigma_F = offline_Sigma_F[0]
offline_T0, offline_T1, offline_T2, offline_T3 = offline_GP_stats[0]
del offline_GP_stats

offline_mean, _, _, _ = BI.prior_mniw_2naturalPara_inv(
    GP_prior[0] + np.mean(offline_T0, axis=0),
    GP_prior[1] + np.mean(offline_T1, axis=0),
    GP_prior[2] + np.mean(offline_T2, axis=0),
    GP_prior[3] + np.mean(offline_T3, axis=0),
)


################################################################################
### Offline Algorithm (PGAS)
print("\n=== Offline Algorithm (PGAS) ===")

# Run PGAS
(
    offline_Sigma_X_PGAS,
    offline_log_likelihood_PGAS,
) = EMPS_PGAS_baseline(key, init_ref_state)

# Get sufficient statistics
basis = jax.vmap(jax.vmap(basis_fcn_f_PGAS, in_axes=(0, None)))(
    offline_Sigma_X_PGAS[:-1], ctrl_input[:-1]
)
T = jax.vmap(jax.vmap(BI.prior_mniw_calcStatistics))(
    offline_Sigma_X_PGAS[1:], basis
)
PGAS_Posterior_Stats = (
    GP_prior_PGAS[0] + np.mean(np.sum(T[0], axis=0), axis=0),
    GP_prior_PGAS[1] + np.mean(np.sum(T[1], axis=0), axis=0),
    GP_prior_PGAS[2] + np.mean(np.sum(T[2], axis=0), axis=0),
    GP_prior_PGAS[3] + np.mean(np.sum(T[3], axis=0), axis=0),
)
PGAS_mean, _, _, _ = BI.prior_mniw_2naturalPara_inv(*PGAS_Posterior_Stats)

# Run Validation Simulation
RMSE_Alg2, RMSE_PGAS = EMPS_Validation_Simulation(offline_mean, PGAS_mean)
print(f"RMSE_Alg2: {RMSE_Alg2}")
print(f"RMSE_PGAS: {RMSE_PGAS}")

################################################################################
# Saving Results

dq_plot = jnp.linspace(-0.15, 0.15, 500)
basis_plot = jax.vmap(basis_fcn)(dq_plot)

# Create save file
mdict = {
    "offline_Sigma_X": offline_Sigma_X,
    "offline_Sigma_F": offline_Sigma_F,
    "offline_Sigma_Y": offline_Sigma_Y,
    "offline_weights": offline_weights,
    "offline_log_likelihood": offline_log_likelihood,
    "offline_T0": offline_T0,
    "offline_T1": offline_T1,
    "offline_T2": offline_T2,
    "offline_T3": offline_T3,
    "online_Sigma_X": online_Sigma_X,
    "online_Sigma_F": online_Sigma_F,
    "online_Sigma_Y": online_Sigma_Y,
    "online_weights": online_weights,
    "online_log_likelihood": online_log_likelihood,
    "online_T0": online_T0,
    "online_T1": online_T1,
    "online_T2": online_T2,
    "online_T3": online_T3,
    "offline_Sigma_X_PGAS": offline_Sigma_X_PGAS,
    "offline_log_likelihood_PGAS": offline_log_likelihood_PGAS,
    "time": time,
    "dq_plot": dq_plot,
    "basis_plot": basis_plot,
    "prior_T0": GP_prior[0],
    "prior_T1": GP_prior[1],
    "prior_T2": GP_prior[2],
    "prior_T3": GP_prior[3],
    "RMSE_Alg2": RMSE_Alg2,
    "RMSE_PGAS": RMSE_PGAS,
    "Y": Y,
    "X": X,
}
scipy.io.savemat("plots/EMPS.mat", mdict)
