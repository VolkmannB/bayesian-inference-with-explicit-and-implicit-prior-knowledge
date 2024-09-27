import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import scipy.io

import matplotlib.pyplot as plt

from src.Vehicle import time, steps, GP_prior_f
from src.Publication_Plotting import plot_Data, apply_basic_formatting
from src.Publication_Plotting import plot_fcn_error_1D, imes_blue, imes_orange
from src.BayesianInferrence import prior_mniw_Predictive, prior_mniw_2naturalPara_inv



################################################################################


N_slices = 2


data = scipy.io.loadmat('plots\Vehicle.mat')


offline_Sigma_X = data['offline_Sigma_X']
offline_Sigma_mu_f = data['offline_Sigma_mu_f']
offline_Sigma_mu_r = data['offline_Sigma_mu_r']
offline_Sigma_alpha_f = data['offline_Sigma_alpha_f']
offline_Sigma_alpha_r = data['offline_Sigma_alpha_r']
offline_weights = data['offline_weights']
offline_T0_f = data['offline_T0_f']
offline_T1_f = data['offline_T1_f']
offline_T2_f = data['offline_T2_f']
offline_T3_f = data['offline_T3_f'].flatten()
offline_T0_r = data['offline_T0_r']
offline_T1_r = data['offline_T1_r']
offline_T2_r = data['offline_T2_r']
offline_T3_r = data['offline_T3_r'].flatten()

online_Sigma_X = data['online_Sigma_X']
online_Sigma_mu_f = data['online_Sigma_mu_f']
online_Sigma_mu_r = data['online_Sigma_mu_r']
online_Sigma_alpha_f = data['online_Sigma_alpha_f']
online_Sigma_alpha_r = data['online_Sigma_alpha_r']
online_weights = data['online_weights']
online_T0_f = data['online_T0_f']
online_T1_f = data['online_T1_f']
online_T2_f = data['online_T2_f']
online_T3_f = data['online_T3_f'].flatten()
online_T0_r = data['online_T0_r']
online_T1_r = data['online_T1_r']
online_T2_r = data['online_T2_r']
online_T3_r = data['online_T3_r'].flatten()

GP_prior_stats_f = [data['prior_T0_f'], data['prior_T1_f'], 
                  data['prior_T2_f'], data['prior_T3_f'].flatten()]
GP_prior_stats_r = [data['prior_T0_r'], data['prior_T1_r'], 
                  data['prior_T2_r'], data['prior_T3_r'].flatten()]

alpha_plot = data['alpha_plot'].flatten()
basis_plot = data['basis_plot']
mu_true_plot = data['mu_true_plot'].flatten()

X = data['X']
mu_f = data['mu_f'].flatten()
mu_r = data['mu_r'].flatten()

del data
    
    
    
# Convert sufficient statistics to standard parameters
(offline_GP_Mean_f, offline_GP_Col_Cov_f, 
 offline_GP_Row_Scale_f, offline_GP_df_f) = jax.vmap(prior_mniw_2naturalPara_inv)(
    GP_prior_stats_f[0] + np.cumsum(offline_T0_f, axis=0),
    GP_prior_stats_f[1] + np.cumsum(offline_T1_f, axis=0),
    GP_prior_stats_f[2] + np.cumsum(offline_T2_f, axis=0),
    GP_prior_stats_f[3] + np.cumsum(offline_T3_f, axis=0)
)
(offline_GP_Mean_r, offline_GP_Col_Cov_r, 
 offline_GP_Row_Scale_r, offline_GP_df_r) = jax.vmap(prior_mniw_2naturalPara_inv)(
    GP_prior_stats_r[0] + np.cumsum(offline_T0_r, axis=0),
    GP_prior_stats_r[1] + np.cumsum(offline_T1_r, axis=0),
    GP_prior_stats_r[2] + np.cumsum(offline_T2_r, axis=0),
    GP_prior_stats_r[3] + np.cumsum(offline_T3_r, axis=0)
)
del offline_T0_f, offline_T1_f, offline_T2_f, offline_T3_f
del offline_T0_r, offline_T1_r, offline_T2_r, offline_T3_r


# Convert sufficient statistics to standard parameters
(online_GP_Mean_f, online_GP_Col_Cov_f, 
 online_GP_Row_Scale_f, online_GP_df_f) = jax.vmap(prior_mniw_2naturalPara_inv)(
    GP_prior_stats_f[0] + online_T0_f,
    GP_prior_stats_f[1] + online_T1_f,
    GP_prior_stats_f[2] + online_T2_f,
    GP_prior_stats_f[3] + online_T3_f
)
(online_GP_Mean_r, online_GP_Col_Cov_r, 
 online_GP_Row_Scale_r, online_GP_df_r) = jax.vmap(prior_mniw_2naturalPara_inv)(
    GP_prior_stats_r[0] + online_T0_r,
    GP_prior_stats_r[1] + online_T1_r,
    GP_prior_stats_r[2] + online_T2_r,
    GP_prior_stats_r[3] + online_T3_r
)
del online_T0_f, online_T1_f, online_T2_f, online_T3_f
del online_T0_r, online_T1_r, online_T2_r, online_T3_r


# function values with GP prior
GP_prior_f = prior_mniw_2naturalPara_inv(
            GP_prior_stats_f[0],
            GP_prior_stats_f[1],
            GP_prior_stats_f[2],
            GP_prior_stats_f[3]
        )
_, col_scale_prior, row_scale_prior, _ = prior_mniw_Predictive(
    mean=GP_prior_f[0], 
    col_cov=GP_prior_f[1], 
    row_scale=GP_prior_f[2], 
    df=GP_prior_f[3], 
    basis=basis_plot)
fcn_var_prior = np.diag(col_scale_prior-1) * row_scale_prior[0,0]
del col_scale_prior, row_scale_prior, GP_prior_f



################################################################################
# Plotting Offline


# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=np.concatenate([offline_Sigma_X, offline_Sigma_mu_f[...,None], offline_Sigma_mu_r[...,None]], axis=-1),
    weights=offline_weights,
    Reference=np.concatenate([X, mu_f[...,None], mu_r[...,None]], axis=-1),
    time=time
)
axes_X[0].set_ylabel(r"$\psi$ in $\mathrm{rad/s}$")
axes_X[1].set_ylabel(r"$v_y$ in $\mathrm{m/s}$")
axes_X[2].set_ylabel(r"$\mu_\mathrm{f}$")
axes_X[3].set_ylabel(r"$\mu_\mathrm{r}$")
axes_X[3].set_xlabel(r"Time in $\mathrm{s}$")
axes_X[2].set_ylim(-2,2)
axes_X[3].set_ylim(-2,2)
apply_basic_formatting(fig_X, width=10, aspect_ratio=0.6, font_size=8)
fig_X.savefig("plots\Vehicle_PGAS_X.svg", bbox_inches='tight')

N_PGAS_iter = offline_Sigma_X.shape[1]
index = (np.array(range(N_slices))+1)/N_slices*(N_PGAS_iter-1)



# function value from GP
fcn_mean_f = np.zeros((N_PGAS_iter, alpha_plot.shape[0]))
fcn_var_f = np.zeros((N_PGAS_iter, alpha_plot.shape[0]))
fcn_mean_r = np.zeros((N_PGAS_iter, alpha_plot.shape[0]))
fcn_var_r = np.zeros((N_PGAS_iter, alpha_plot.shape[0]))
for i in tqdm(range(0, N_PGAS_iter), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=offline_GP_Mean_f[i], 
        col_cov=offline_GP_Col_Cov_f[i], 
        row_scale=offline_GP_Row_Scale_f[i], 
        df=offline_GP_df_f[i], 
        basis=basis_plot)
    fcn_var_f[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_f[i] = mean
    
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=offline_GP_Mean_r[i], 
        col_cov=offline_GP_Col_Cov_r[i], 
        row_scale=offline_GP_Row_Scale_r[i], 
        df=offline_GP_df_r[i], 
        basis=basis_plot)
    fcn_var_r[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_r[i] = mean

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        alpha_plot, 
        Mean=fcn_mean_f[int(i)], 
        Std=np.sqrt(fcn_var_f[int(i)]),
        X_stats=offline_Sigma_alpha_f[:,:int(i)], 
        X_weights=offline_weights[:,:int(i)])
    ax_fcn_e[0][-1].set_xlabel(r"$\alpha$ in $\mathrm{rad}$")
    ax_fcn_e[0][-1].plot(alpha_plot, mu_true_plot, color='red', linestyle=':')
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"plots\Vehicle_PGAS_muf_fcn_{int(i)}.svg")



# plot weighted RMSE of GP over entire function space
w = 1 / fcn_var_f
w = w / np.sum(w, axis=-1, keepdims=True)
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE_f = np.sqrt(1/(v1-(v2/v1**2)) * jnp.sum((fcn_mean_f - mu_true_plot)**2 * w, axis=-1))

w = 1 / fcn_var_r
w = w / np.sum(w, axis=-1, keepdims=True)
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE_r = np.sqrt(1/(v1-(v2/v1**2)) * jnp.sum((fcn_mean_r - mu_true_plot)**2 * w, axis=-1))

fig_RMSE, ax_RMSE = plt.subplots(1,1, layout='tight')
ax_RMSE.plot(
    range(0,N_PGAS_iter),
    wRMSE_f,
    color=imes_blue
)
ax_RMSE.plot(
    range(0,N_PGAS_iter),
    wRMSE_r,
    color=imes_orange
)
ax_RMSE.set_ylabel(r"wRMSE")
ax_RMSE.set_xlabel(r"Time in $\mathrm{s}$")
ax_RMSE.set_ylim(0)

for i in index:
    ax_RMSE.plot([int(i), int(i)], [0, wRMSE_f[int(i)]*1.5], color="black", linewidth=0.8)
    
wRMSE_offline_f = wRMSE_f[-1]
wRMSE_offline_r = wRMSE_r[-1]
    
    
apply_basic_formatting(fig_RMSE, width=8, font_size=8)
fig_RMSE.savefig("plots\Vehicle_PGAS_muf_wRMSE.svg", bbox_inches='tight')



################################################################################
# Plotting Online


# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=np.concatenate([online_Sigma_X, online_Sigma_mu_f[...,None], online_Sigma_mu_r[...,None]], axis=-1),
    weights=online_weights,
    Reference=np.concatenate([X, mu_f[...,None], mu_r[...,None]], axis=-1),
    time=time
)
axes_X[0].set_ylabel(r"$\psi$ in $\mathrm{rad/s}$")
axes_X[1].set_ylabel(r"$v_y$ in $\mathrm{m/s}$")
axes_X[2].set_ylabel(r"$\mu_\mathrm{f}$")
axes_X[3].set_ylabel(r"$\mu_\mathrm{r}$")
axes_X[3].set_xlabel(r"Time in $\mathrm{s}$")
axes_X[2].set_ylim(-2,2)
axes_X[3].set_ylim(-2,2)
apply_basic_formatting(fig_X, width=10, aspect_ratio=0.6, font_size=8)
fig_X.savefig("plots\Vehicle_APF_X.svg", bbox_inches='tight')

steps = time.shape[0]
index = (np.array(range(N_slices))+1)/N_slices*(steps-1)



# function value from GP
fcn_mean_f = np.zeros((steps, alpha_plot.shape[0]))
fcn_var_f = np.zeros((steps, alpha_plot.shape[0]))
fcn_mean_r = np.zeros((steps, alpha_plot.shape[0]))
fcn_var_r = np.zeros((steps, alpha_plot.shape[0]))
for i in tqdm(range(0, steps), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=online_GP_Mean_f[i], 
        col_cov=online_GP_Col_Cov_f[i], 
        row_scale=online_GP_Row_Scale_f[i], 
        df=online_GP_df_f[i], 
        basis=basis_plot)
    fcn_var_f[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_f[i] = mean
    
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=online_GP_Mean_r[i], 
        col_cov=online_GP_Col_Cov_r[i], 
        row_scale=online_GP_Row_Scale_r[i], 
        df=online_GP_df_r[i], 
        basis=basis_plot)
    fcn_var_r[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_r[i] = mean

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        alpha_plot, 
        Mean=fcn_mean_f[int(i)], 
        Std=np.sqrt(fcn_var_f[int(i)]),
        X_stats=online_Sigma_alpha_f[:int(i)], 
        X_weights=online_weights[:int(i)])
    ax_fcn_e[0][-1].set_xlabel(r"$\alpha$ in $\mathrm{rad}$")
    ax_fcn_e[0][-1].plot(alpha_plot, mu_true_plot, color='red', linestyle=':')
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"plots\Vehicle_APF_muf_fcn_{np.round(time[int(i)],3)}.svg")



# plot weighted RMSE of GP over entire function space
w = 1 / fcn_var_f
w = w / np.sum(w, axis=-1, keepdims=True)
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE_f = np.sqrt(1/(v1-(v2/v1**2)) * jnp.sum((fcn_mean_f - mu_true_plot)**2 * w, axis=-1))

w = 1 / fcn_var_r
w = w / np.sum(w, axis=-1, keepdims=True)
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE_r = np.sqrt(1/(v1-(v2/v1**2)) * jnp.sum((fcn_mean_r - mu_true_plot)**2 * w, axis=-1))

fig_RMSE, ax_RMSE = plt.subplots(1,1, layout='tight')
ax_RMSE.plot(
    time,
    wRMSE_f,
    color=imes_blue
)
ax_RMSE.plot(
    time,
    wRMSE_r,
    color=imes_orange
)
ax_RMSE.set_ylabel(r"wRMSE")
ax_RMSE.set_xlabel(r"Time in $\mathrm{s}$")
ax_RMSE.set_ylim(0)

# plot convergence wRMSE of offline algorithm
ax_RMSE.plot(
    [time[0], time[-1]], 
    [wRMSE_offline_f, wRMSE_offline_f], 
    color='red', linestyle=':')

for i in index:
    ax_RMSE.plot([time[int(i)], time[int(i)]], [0, wRMSE_f[int(i)]*1.5], color="black", linewidth=0.8)
    
    
apply_basic_formatting(fig_RMSE, width=8, font_size=8)
fig_RMSE.savefig("plots\Vehicle_APF_muf_wRMSE.svg", bbox_inches='tight')



plt.show()