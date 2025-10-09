import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.Filtering import reconstruct_trajectory
from src.Toy_Example import (
    GP_prior,
    X,
    Toy_Example_Algorithm1,
    Toy_Example_Algorithm2,
    Toy_Example_PGAS,
    basis_fcn,
    f_x,
    key,
)
import src.BayesianInferrence as BI

################################################################################
# Online Algorithm
print("\n=== Online Algorithm ===")

key, key_sim = jax.random.split(key)
(
    online_Sigma_X,
    online_Sigma_xi,
    online_GP_stats,
    online_weights,
    _,
    _,
    online_Sigma_Y,
    online_log_likelihood,
) = Toy_Example_Algorithm1(key_sim)
online_Sigma_xi = online_Sigma_xi[0]
online_T0, online_T1, online_T2, online_T3 = online_GP_stats[0]
del online_GP_stats

################################################################################
# Offline Algorithm
print("\n=== Offline Algorithm ===")

# Create reference
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
) = Toy_Example_Algorithm1(key_sim)
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
    offline_Sigma_xi,
    offline_weights,
    offline_GP_stats,
    offline_Sigma_Y,
    offline_log_likelihood,
) = Toy_Example_Algorithm2(key, init_ref_state[:, None], init_ref_int_var)
offline_Sigma_xi = offline_Sigma_xi[0]
offline_T0, offline_T1, offline_T2, offline_T3 = offline_GP_stats[0]
del offline_GP_stats

################################################################################
# Offline Algorithm (PGAS)
print("\n=== Offline Algorithm (PGAS) ===")

# Run PGAS
(
    pgas_Sigma_X,
    pgas_log_likelihood,
) = Toy_Example_PGAS(key, init_ref_state)

# Get sufficient statistics
basis = jax.vmap(jax.vmap(basis_fcn))(pgas_Sigma_X[:-1])
T = jax.vmap(jax.vmap(BI.prior_mniw_calcStatistics))(pgas_Sigma_X[1:], basis)
pgas_T0 = np.mean(np.sum(T[0], axis=0), axis=0)
pgas_T1 = np.mean(np.sum(T[1], axis=0), axis=0)
pgas_T2 = np.mean(np.sum(T[2], axis=0), axis=0)
pgas_T3 = np.mean(np.sum(T[3], axis=0), axis=0)

################################################################################
# Save Results

# precompute true function for later plotting
x_plot = jnp.linspace(-30, 30, 500)
fx_true_plot = jax.vmap(f_x)(x_plot)
basis_plot = jax.vmap(basis_fcn)(x_plot)

# Calculate system model from Algorithm 1
online_suff_stats = (
    GP_prior[0] + online_T0[-1],
    GP_prior[1] + online_T1[-1],
    GP_prior[2] + online_T2[-1],
    GP_prior[3] + online_T3[-1],
)
online_std_params = BI.prior_mniw_2naturalPara_inv(*online_suff_stats)
online_fcn_mean, online_col_scale, online_row_scale, _ = (
    BI.prior_mniw_Predictive(
        mean=online_std_params[0],
        col_cov=online_std_params[1],
        row_scale=online_std_params[2],
        df=online_std_params[3],
        basis=basis_plot,
    )
)
online_fcn_var = np.diag(online_col_scale - 1) * online_row_scale[0, 0]


# Calculate system model from Algorithm 2
offline_suff_stats = (
    GP_prior[0] + np.mean(offline_T0, axis=0),
    GP_prior[1] + np.mean(offline_T1, axis=0),
    GP_prior[2] + np.mean(offline_T2, axis=0),
    GP_prior[3] + np.mean(offline_T3, axis=0),
)
offline_std_params = BI.prior_mniw_2naturalPara_inv(*offline_suff_stats)
offline_fcn_mean, offline_col_scale, offline_row_scale, _ = (
    BI.prior_mniw_Predictive(
        mean=offline_std_params[0],
        col_cov=offline_std_params[1],
        row_scale=offline_std_params[2],
        df=offline_std_params[3],
        basis=basis_plot,
    )
)
offline_fcn_var = np.diag(offline_col_scale - 1) * offline_row_scale[0, 0]


# Calculate system model from PGAS
pgas_suff_stats = (
    GP_prior[0] + pgas_T0,
    GP_prior[1] + pgas_T1,
    GP_prior[2] + pgas_T2,
    GP_prior[3] + pgas_T3,
)
pgas_std_params = BI.prior_mniw_2naturalPara_inv(*pgas_suff_stats)
pgas_fcn_mean, pgas_col_scale, pgas_row_scale, _ = BI.prior_mniw_Predictive(
    mean=pgas_std_params[0],
    col_cov=pgas_std_params[1],
    row_scale=pgas_std_params[2],
    df=pgas_std_params[3],
    basis=basis_plot,
)
pgas_fcn_var = np.diag(pgas_col_scale - 1) * pgas_row_scale[0, 0]

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot data
ax.scatter(X[0:-1], X[1:], s=10, alpha=0.5, label="True State", color="blue")
# ax.scatter(
#     X[0:-1], Y[1:], s=10, alpha=0.5, label="Measurements", color="orange"
# )

# Plot true function
ax.plot(x_plot, fx_true_plot, "r--", label="True function")

# Plot Online (Algorithm 1) estimate
ax.plot(
    x_plot,
    online_fcn_mean.flatten(),
    label="Online estimate (Algorithm 1)",
    color="green",
)
ax.fill_between(
    x_plot,
    (online_fcn_mean - 3 * np.sqrt(online_fcn_var)).flatten(),
    (online_fcn_mean + 3 * np.sqrt(online_fcn_var)).flatten(),
    alpha=0.2,
    color="green",
)

# Plot Offline (Algorithm 2) estimate
ax.plot(
    x_plot,
    offline_fcn_mean.flatten(),
    label="Offline estimate (Algorithm 2)",
    color="blue",
)
ax.fill_between(
    x_plot,
    (offline_fcn_mean - 3 * np.sqrt(offline_fcn_var)).flatten(),
    (offline_fcn_mean + 3 * np.sqrt(offline_fcn_var)).flatten(),
    alpha=0.2,
    color="blue",
)

# Plot PGAS estimate
ax.plot(
    x_plot,
    pgas_fcn_mean.flatten(),
    label="PGAS estimate",
    color="cyan",
)
ax.fill_between(
    x_plot,
    (pgas_fcn_mean - 3 * np.sqrt(pgas_fcn_var)).flatten(),
    (pgas_fcn_mean + 3 * np.sqrt(pgas_fcn_var)).flatten(),
    alpha=0.2,
    color="cyan",
)

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Toy Example Simulation Results")
ax.legend()
ax.grid(True)
ax.set_ylim(-20, 20)

fig.savefig("plots/Toy_Example_estimates.pdf", bbox_inches="tight")
plt.show()
