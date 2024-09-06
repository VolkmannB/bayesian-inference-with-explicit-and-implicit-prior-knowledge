import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import scipy.io

import matplotlib.pyplot as plt

from src.Publication_Plotting import plot_fcn_error_2D
from src.Publication_Plotting import plot_Data, apply_basic_formatting, imes_blue
from src.BayesianInferrence import prior_mniw_Predictive, prior_mniw_2naturalPara_inv



################################################################################

N_slices = 2

data = scipy.io.loadmat('plots\SingleMassOscillator.mat')

offline_Sigma_X = data['offline_Sigma_X']
offline_Sigma_F = data['offline_Sigma_F']
offline_weights = data['offline_weights']
offline_T0 = data['offline_T0']
offline_T1 = data['offline_T1']
offline_T2 = data['offline_T2']
offline_T3 = data['offline_T3'].flatten()
online_Sigma_X = data['online_Sigma_X']
online_Sigma_F = data['online_Sigma_F']
online_weights = data['online_weights']
online_T0 = data['online_T0']
online_T1 = data['online_T1']
online_T2 = data['online_T2']
online_T3 = data['online_T3'].flatten()
time = data['time'].flatten()
X_plot = data['X_plot']
basis_plot = data['basis_plot']
F_sd_true_plot = data['F_sd_true_plot'].flatten()
GP_prior_stats = [data['prior_T0'], data['prior_T1'], 
            data['prior_T2'], data['prior_T3'].flatten()]
X = data['X']
F_sd = data['F_sd'].flatten()

del data
    
    
# Convert sufficient statistics to standard parameters
(offline_GP_Mean, offline_GP_Col_Cov, 
 offline_GP_Row_Scale, offline_GP_df) = jax.vmap(prior_mniw_2naturalPara_inv)(
    GP_prior_stats[0] + np.cumsum(offline_T0, axis=0),
    GP_prior_stats[1] + np.cumsum(offline_T1, axis=0),
    GP_prior_stats[2] + np.cumsum(offline_T2, axis=0),
    GP_prior_stats[3] + np.cumsum(offline_T3, axis=0)
)
    
    
# Convert sufficient statistics to standard parameters
(online_GP_Mean, online_GP_Col_Cov, 
 online_GP_Row_Scale, online_GP_df) = jax.vmap(prior_mniw_2naturalPara_inv)(
    GP_prior_stats[0] + online_T0,
    GP_prior_stats[1] + online_T1,
    GP_prior_stats[2] + online_T2,
    GP_prior_stats[3] + online_T3
 )

# function values with GP prior
GP_prior = prior_mniw_2naturalPara_inv(
            GP_prior_stats[0],
            GP_prior_stats[1],
            GP_prior_stats[2],
            GP_prior_stats[3]
        )
_, col_scale_prior, row_scale_prior, _ = prior_mniw_Predictive(
    mean=GP_prior[0], 
    col_cov=GP_prior[1], 
    row_scale=GP_prior[2], 
    df=GP_prior[3], 
    basis=basis_plot)
fcn_var_prior = np.diag(col_scale_prior-1) * row_scale_prior[0,0]
del col_scale_prior, row_scale_prior, GP_prior

    

################################################################################
# Plots Offline


# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=np.concatenate([offline_Sigma_X, offline_Sigma_F[...,None]], axis=-1),
    weights=offline_weights,
    Reference=np.concatenate([X,F_sd[...,None]], axis=-1),
    time=time
)
axes_X[0].set_ylabel(r"$s$ in $\mathrm{m}$")
axes_X[1].set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
axes_X[2].set_ylabel(r"$F$ in $\mathrm{N}$")
axes_X[2].set_xlabel(r"Time in $\mathrm{s}$")
apply_basic_formatting(fig_X, width=10, aspect_ratio=0.6, font_size=8)
fig_X.savefig("plots\SingleMassOscillator_PGAS_X.svg", bbox_inches='tight')

N_iterations = offline_Sigma_X.shape[1]
index = (np.array(range(N_slices))+1)/N_slices*(N_iterations-1)

# function value from GP
fcn_mean = np.zeros((N_iterations, X_plot.shape[0]))
fcn_var = np.zeros((N_iterations, X_plot.shape[0]))
for i in tqdm(range(0, N_iterations), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=offline_GP_Mean[i], 
        col_cov=offline_GP_Col_Cov[i], 
        row_scale=offline_GP_Row_Scale[i], 
        df=offline_GP_df[i], 
        basis=basis_plot)
    fcn_var[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean[i] = mean

# normalize variance to create transparency effect
fcn_alpha = np.maximum(np.minimum(1 - fcn_var/fcn_var_prior, 1), 0)

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_2D(
        X_plot, 
        Mean=np.abs(fcn_mean[int(i)]-F_sd_true_plot), 
        X_stats=offline_Sigma_X[:,:int(i)], 
        X_weights=offline_weights[:,:int(i)], 
        alpha=fcn_alpha[int(i)])
    ax_fcn_e[0].set_xlabel(r"$s$ in $\mathrm{m}$")
    ax_fcn_e[0].set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"plots\SingleMassOscillator_PGAS_Fsd_fcn_{int(i)}.svg")



# plot weighted RMSE of GP over entire function space
w = 1 / fcn_var
w = w / np.sum(w, axis=-1, keepdims=True)
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE = np.sqrt(1/(v1-(v2/v1**2)) * jnp.sum((fcn_mean - F_sd_true_plot)**2 * w, axis=-1))
fig_RMSE, ax_RMSE = plt.subplots(1,1, layout='tight')
ax_RMSE.plot(
    range(0,N_iterations),
    wRMSE,
    color=imes_blue
)
ax_RMSE.set_ylabel(r"wRMSE")
ax_RMSE.set_xlabel(r"Iterations")
ax_RMSE.set_ylim(0)

for i in index:
    ax_RMSE.plot([int(i), int(i)], [0, wRMSE[int(i)]*1.5], color="black", linewidth=0.8)
    
wRMSE_offline_final = wRMSE[-1]
    
apply_basic_formatting(fig_RMSE, width=8, font_size=8)
fig_RMSE.savefig("plots\SingleMassOscillator_PGAS_Fsd_wRMSE.svg", bbox_inches='tight')



################################################################################
# Plots Online



# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=np.concatenate([online_Sigma_X, online_Sigma_F[...,None]], axis=-1),
    weights=online_weights,
    Reference=np.concatenate([X,F_sd[...,None]], axis=-1),
    time=time
)
axes_X[0].set_ylabel(r"$s$ in $\mathrm{m}$")
axes_X[1].set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
axes_X[2].set_ylabel(r"$F$ in $\mathrm{N}$")
axes_X[2].set_xlabel(r"Time in $\mathrm{s}$")
apply_basic_formatting(fig_X, width=10, aspect_ratio=0.6, font_size=8)
fig_X.savefig("plots\SingleMassOscillator_APF_X.svg", bbox_inches='tight')

index = (np.array(range(N_slices))+1)/N_slices*(time.shape[0]-1)

# function value from GP
fcn_mean = np.zeros((time.shape[0], X_plot.shape[0]))
fcn_var = np.zeros((time.shape[0], X_plot.shape[0]))
for i in tqdm(range(0, time.shape[0]), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=online_GP_Mean[i], 
        col_cov=online_GP_Col_Cov[i], 
        row_scale=online_GP_Row_Scale[i], 
        df=online_GP_df[i], 
        basis=basis_plot)
    fcn_var[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean[i] = mean

# normalize variance to create transparency effect
fcn_alpha = np.maximum(np.minimum(1 - fcn_var/fcn_var_prior, 1), 0)

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_2D(
        X_plot, 
        Mean=np.abs(fcn_mean[int(i)]-F_sd_true_plot), 
        X_stats=online_Sigma_X[:int(i)], 
        X_weights=online_weights[:int(i)], 
        alpha=fcn_alpha[int(i)])
    ax_fcn_e[0].set_xlabel(r"$s$ in $\mathrm{m}$")
    ax_fcn_e[0].set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"plots\SingleMassOscillator_APF_Fsd_fcn_{np.round(time[int(i)],3)}.svg")



# plot weighted RMSE of GP over entire function space
w = 1 / fcn_var
w = w / np.sum(w, axis=-1, keepdims=True)
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE = np.sqrt(1/(v1-(v2/v1**2)) * jnp.sum((fcn_mean - F_sd_true_plot)**2 * w, axis=-1))
fig_RMSE, ax_RMSE = plt.subplots(1,1, layout='tight')
ax_RMSE.plot(
    time,
    wRMSE,
    color=imes_blue
)
ax_RMSE.set_ylabel(r"wRMSE")
ax_RMSE.set_xlabel(r"Time in $\mathrm{s}$")
ax_RMSE.set_ylim(0)

# plot convergence wRMSE of offline algorithm
ax_RMSE.plot(
    [time[0], time[-1]], 
    [wRMSE_offline_final, wRMSE_offline_final], 
    color='red', linestyle=':')

for i in index:
    ax_RMSE.plot([time[int(i)], time[int(i)]], [0, wRMSE[int(i)]*1.5], color="black", linewidth=0.8)
    
apply_basic_formatting(fig_RMSE, width=8, font_size=8)
fig_RMSE.savefig("plots\SingleMassOscillator_APF_Fsd_wRMSE.svg", bbox_inches='tight')



plt.show()