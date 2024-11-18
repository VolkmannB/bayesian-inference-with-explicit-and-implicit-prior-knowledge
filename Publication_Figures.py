import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.gridspec

from src.Publication_Plotting import plot_fcn_error_2D, plot_fcn_error_1D
from src.Publication_Plotting import plot_Data, apply_basic_formatting
from src.Publication_Plotting import imes_blue, imes_orange, calc_wRMSE
from src.BayesianInferrence import prior_mniw_Predictive
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.EMPS import EMPS_Validation_Simulation



PGAS_slice_idx = [49, 799]
APF_slice_idx = [499, -1]
APF_slice1 = 0.2


################################################################################
# Load SMO Data

print('Loading SMO data')
data = scipy.io.loadmat('plots\SingleMassOscillator.mat')

SMO_offline_Sigma_X = data['offline_Sigma_X']
SMO_offline_Sigma_F = data['offline_Sigma_F']
SMO_offline_weights = data['offline_weights']
SMO_offline_T0 = data['offline_T0']
SMO_offline_T1 = data['offline_T1']
SMO_offline_T2 = data['offline_T2']
SMO_offline_T3 = data['offline_T3'].flatten()
SMO_online_Sigma_X = data['online_Sigma_X']
SMO_online_Sigma_F = data['online_Sigma_F']
SMO_online_weights = data['online_weights']
SMO_online_T0 = data['online_T0']
SMO_online_T1 = data['online_T1']
SMO_online_T2 = data['online_T2']
SMO_online_T3 = data['online_T3'].flatten()
SMO_time = data['time'].flatten()
SMO_X_plot = data['X_plot']
SMO_basis_plot = data['basis_plot']
SMO_F_sd_true_plot = data['F_sd_true_plot'].flatten()
SMO_GP_prior_stats = [data['prior_T0'], data['prior_T1'], 
            data['prior_T2'], data['prior_T3'].flatten()]
SMO_X = data['X']
SMO_F_sd = data['F_sd'].flatten()

del data
    
    
# Convert sufficient statistics to standard parameters
(SMO_offline_GP_Mean, SMO_offline_GP_Col_Cov, 
 SMO_offline_GP_Row_Scale, SMO_offline_GP_df) = jax.vmap(prior_mniw_2naturalPara_inv)(
    SMO_GP_prior_stats[0] + np.cumsum(SMO_offline_T0, axis=0)/np.arange(1,SMO_offline_Sigma_X.shape[1]+1)[:,None,None],
    SMO_GP_prior_stats[1] + np.cumsum(SMO_offline_T1, axis=0)/np.arange(1,SMO_offline_Sigma_X.shape[1]+1)[:,None,None],
    SMO_GP_prior_stats[2] + np.cumsum(SMO_offline_T2, axis=0)/np.arange(1,SMO_offline_Sigma_X.shape[1]+1)[:,None,None],
    SMO_GP_prior_stats[3] + np.cumsum(SMO_offline_T3, axis=0)/np.arange(1,SMO_offline_Sigma_X.shape[1]+1)
)
    
    
# Convert sufficient statistics to standard parameters
(SMO_online_GP_Mean, SMO_online_GP_Col_Cov, 
 SMO_online_GP_Row_Scale, SMO_online_GP_df) = jax.vmap(prior_mniw_2naturalPara_inv)(
    SMO_GP_prior_stats[0] + SMO_online_T0,
    SMO_GP_prior_stats[1] + SMO_online_T1,
    SMO_GP_prior_stats[2] + SMO_online_T2,
    SMO_GP_prior_stats[3] + SMO_online_T3
)

# function values with GP prior
SMO_GP_prior = prior_mniw_2naturalPara_inv(
            SMO_GP_prior_stats[0],
            SMO_GP_prior_stats[1],
            SMO_GP_prior_stats[2],
            SMO_GP_prior_stats[3]
        )
_, col_scale_prior, row_scale_prior, _ = prior_mniw_Predictive(
    mean=SMO_GP_prior[0], 
    col_cov=SMO_GP_prior[1], 
    row_scale=SMO_GP_prior[2], 
    df=SMO_GP_prior[3], 
    basis=SMO_basis_plot)
SMO_fcn_var_prior = np.diag(col_scale_prior-1) * row_scale_prior[0,0]
del col_scale_prior, row_scale_prior, SMO_GP_prior


APF_SMO_slice_idx1 = (np.abs(SMO_time - SMO_time[-1]*APF_slice1)).argmin()

    

################################################################################
# Load Vehicle Data

print('Loading Vehicle data')
data = scipy.io.loadmat('plots\Vehicle.mat')


Veh_offline_Sigma_X = data['offline_Sigma_X']
Veh_offline_Sigma_mu_f = data['offline_Sigma_mu_f']
Veh_offline_Sigma_mu_r = data['offline_Sigma_mu_r']
Veh_offline_Sigma_alpha_f = data['offline_Sigma_alpha_f']
Veh_offline_Sigma_alpha_r = data['offline_Sigma_alpha_r']
Veh_offline_weights = data['offline_weights']
Veh_offline_T0_f = data['offline_T0_f']
Veh_offline_T1_f = data['offline_T1_f']
Veh_offline_T2_f = data['offline_T2_f']
Veh_offline_T3_f = data['offline_T3_f'].flatten()
Veh_offline_T0_r = data['offline_T0_r']
Veh_offline_T1_r = data['offline_T1_r']
Veh_offline_T2_r = data['offline_T2_r']
Veh_offline_T3_r = data['offline_T3_r'].flatten()

Veh_online_Sigma_X = data['online_Sigma_X']
Veh_online_Sigma_mu_f = data['online_Sigma_mu_f']
Veh_online_Sigma_mu_r = data['online_Sigma_mu_r']
Veh_online_Sigma_alpha_f = data['online_Sigma_alpha_f']
Veh_online_Sigma_alpha_r = data['online_Sigma_alpha_r']
Veh_online_weights = data['online_weights']
Veh_online_T0_f = data['online_T0_f']
Veh_online_T1_f = data['online_T1_f']
Veh_online_T2_f = data['online_T2_f']
Veh_online_T3_f = data['online_T3_f'].flatten()
Veh_online_T0_r = data['online_T0_r']
Veh_online_T1_r = data['online_T1_r']
Veh_online_T2_r = data['online_T2_r']
Veh_online_T3_r = data['online_T3_r'].flatten()

Veh_GP_prior_stats_f = [data['prior_T0_f'], data['prior_T1_f'], 
                  data['prior_T2_f'], data['prior_T3_f'].flatten()]
Veh_GP_prior_stats_r = [data['prior_T0_r'], data['prior_T1_r'], 
                  data['prior_T2_r'], data['prior_T3_r'].flatten()]

Veh_alpha_plot = data['alpha_plot'].flatten()
Veh_basis_plot = data['basis_plot']
Veh_mu_true_plot = data['mu_true_plot'].flatten()

Veh_X = data['X']
Veh_mu_f = data['mu_f'].flatten()
Veh_mu_r = data['mu_r'].flatten()
Veh_time = data['time'].flatten()

del data
    
    
    
# Convert sufficient statistics to standard parameters
(Veh_offline_GP_Mean_f, Veh_offline_GP_Col_Cov_f, 
 Veh_offline_GP_Row_Scale_f, Veh_offline_GP_df_f) = jax.vmap(prior_mniw_2naturalPara_inv)(
    Veh_GP_prior_stats_f[0] + np.cumsum(Veh_offline_T0_f, axis=0)/np.arange(1,Veh_offline_Sigma_X.shape[1]+1)[:,None,None],
    Veh_GP_prior_stats_f[1] + np.cumsum(Veh_offline_T1_f, axis=0)/np.arange(1,Veh_offline_Sigma_X.shape[1]+1)[:,None,None],
    Veh_GP_prior_stats_f[2] + np.cumsum(Veh_offline_T2_f, axis=0)/np.arange(1,Veh_offline_Sigma_X.shape[1]+1)[:,None,None],
    Veh_GP_prior_stats_f[3] + np.cumsum(Veh_offline_T3_f, axis=0)/np.arange(1,Veh_offline_Sigma_X.shape[1]+1)
)
(Veh_offline_GP_Mean_r, Veh_offline_GP_Col_Cov_r, 
 Veh_offline_GP_Row_Scale_r, Veh_offline_GP_df_r) = jax.vmap(prior_mniw_2naturalPara_inv)(
    Veh_GP_prior_stats_r[0] + np.cumsum(Veh_offline_T0_r, axis=0)/np.arange(1,Veh_offline_Sigma_X.shape[1]+1)[:,None,None],
    Veh_GP_prior_stats_r[1] + np.cumsum(Veh_offline_T1_r, axis=0)/np.arange(1,Veh_offline_Sigma_X.shape[1]+1)[:,None,None],
    Veh_GP_prior_stats_r[2] + np.cumsum(Veh_offline_T2_r, axis=0)/np.arange(1,Veh_offline_Sigma_X.shape[1]+1)[:,None,None],
    Veh_GP_prior_stats_r[3] + np.cumsum(Veh_offline_T3_r, axis=0)/np.arange(1,Veh_offline_Sigma_X.shape[1]+1)
)
del Veh_offline_T0_f, Veh_offline_T1_f, Veh_offline_T2_f, Veh_offline_T3_f
del Veh_offline_T0_r, Veh_offline_T1_r, Veh_offline_T2_r, Veh_offline_T3_r


# Convert sufficient statistics to standard parameters
(Veh_online_GP_Mean_f, Veh_online_GP_Col_Cov_f, 
 Veh_online_GP_Row_Scale_f, Veh_online_GP_df_f) = jax.vmap(prior_mniw_2naturalPara_inv)(
    Veh_GP_prior_stats_f[0] + Veh_online_T0_f,
    Veh_GP_prior_stats_f[1] + Veh_online_T1_f,
    Veh_GP_prior_stats_f[2] + Veh_online_T2_f,
    Veh_GP_prior_stats_f[3] + Veh_online_T3_f
)
(Veh_online_GP_Mean_r, Veh_online_GP_Col_Cov_r, 
 Veh_online_GP_Row_Scale_r, Veh_online_GP_df_r) = jax.vmap(prior_mniw_2naturalPara_inv)(
    Veh_GP_prior_stats_r[0] + Veh_online_T0_r,
    Veh_GP_prior_stats_r[1] + Veh_online_T1_r,
    Veh_GP_prior_stats_r[2] + Veh_online_T2_r,
    Veh_GP_prior_stats_r[3] + Veh_online_T3_r
)
del Veh_online_T0_f, Veh_online_T1_f, Veh_online_T2_f, Veh_online_T3_f
del Veh_online_T0_r, Veh_online_T1_r, Veh_online_T2_r, Veh_online_T3_r


APF_Veh_slice_idx1 = (np.abs(Veh_time - Veh_time[-1]*APF_slice1)).argmin()


################################################################################
# Load EMPS data

print('loading EMPS data')
data = scipy.io.loadmat('plots\EMPS.mat')


EMPS_offline_Sigma_X = data['offline_Sigma_X']
EMPS_offline_Sigma_F = data['offline_Sigma_F']
EMPS_offline_Sigma_Y = data['offline_Sigma_Y']
EMPS_offline_weights = data['offline_weights']
EMPS_offline_T0 = data['offline_T0']
EMPS_offline_T1 = data['offline_T1']
EMPS_offline_T2 = data['offline_T2']
EMPS_offline_T3 = data['offline_T3'].flatten()

EMPS_online_Sigma_X = data['online_Sigma_X']
EMPS_online_Sigma_F = data['online_Sigma_F']
EMPS_online_Sigma_Y = data['online_Sigma_Y']
EMPS_online_weights = data['online_weights']
EMPS_online_T0 = data['online_T0']
EMPS_online_T1 = data['online_T1']
EMPS_online_T2 = data['online_T2']
EMPS_online_T3 = data['online_T3'].flatten()

EMPS_GP_prior_stats = [data['prior_T0'], data['prior_T1'], 
                  data['prior_T2'], data['prior_T3'].flatten()]

EMPS_dq_plot = data['dq_plot'].flatten()
EMPS_basis_plot = data['basis_plot']
EMPS_time = data['time'].flatten()

EMPS_Y = data['Y'].flatten()
EMPS_X = data['X']

del data
    
    
    
# Convert sufficient statistics to standard parameters
(EMPS_offline_GP_Mean, EMPS_offline_GP_Col_Cov, 
 EMPS_offline_GP_Row_Scale, EMPS_offline_GP_df) = jax.vmap(prior_mniw_2naturalPara_inv)(
    EMPS_GP_prior_stats[0] + np.cumsum(EMPS_offline_T0, axis=0)/np.arange(1,EMPS_offline_Sigma_X.shape[1]+1)[:,None,None],
    EMPS_GP_prior_stats[1] + np.cumsum(EMPS_offline_T1, axis=0)/np.arange(1,EMPS_offline_Sigma_X.shape[1]+1)[:,None,None],
    EMPS_GP_prior_stats[2] + np.cumsum(EMPS_offline_T2, axis=0)/np.arange(1,EMPS_offline_Sigma_X.shape[1]+1)[:,None,None],
    EMPS_GP_prior_stats[3] + np.cumsum(EMPS_offline_T3, axis=0)/np.arange(1,EMPS_offline_Sigma_X.shape[1]+1)
)
del EMPS_offline_T0, EMPS_offline_T1, EMPS_offline_T2, EMPS_offline_T3
    
# Convert sufficient statistics to standard parameters
(EMPS_online_GP_Mean, EMPS_online_GP_Col_Cov, 
 EMPS_online_GP_Row_Scale, EMPS_online_GP_df) = jax.vmap(prior_mniw_2naturalPara_inv)(
    EMPS_GP_prior_stats[0] + EMPS_online_T0,
    EMPS_GP_prior_stats[1] + EMPS_online_T1,
    EMPS_GP_prior_stats[2] + EMPS_online_T2,
    EMPS_GP_prior_stats[3] + EMPS_online_T3
)
del EMPS_online_T0, EMPS_online_T1, EMPS_online_T2, EMPS_online_T3


APF_EMPS_slice_idx1 = (np.abs(EMPS_time - EMPS_time[-1]*APF_slice1)).argmin()

EMPS_validation_GP_RMSE, EMPS_validation_lin_RMSE = EMPS_Validation_Simulation(EMPS_offline_GP_Mean[-1])
print(f"For the validation Data of the EMPS the RMSE is {EMPS_validation_GP_RMSE} m by forward simulation using the mean value of the offline model.")
print(f"For the validation Data of the EMPS the RMSE is {EMPS_validation_lin_RMSE} m by forward simulation using the linear model.")


################################################################################
# Offline

### define layout for trajectory plots

# the figure
fig_traj = plt.figure(dpi=150)
gs_traj = fig_traj.add_gridspec(3, 3,  width_ratios=np.ones(3), height_ratios=np.ones(3))

# generate axes
SMO_ax_x0 = fig_traj.add_subplot(gs_traj[0, 0])
SMO_ax_x1 = fig_traj.add_subplot(gs_traj[1, 0])
SMO_ax_F = fig_traj.add_subplot(gs_traj[2, 0])

Veh_ax_x0 = fig_traj.add_subplot(gs_traj[0, 1])
Veh_ax_x1 = fig_traj.add_subplot(gs_traj[1, 1])
Veh_ax_muf = fig_traj.add_subplot(gs_traj[2, 1])

EMPS_ax_x0 = fig_traj.add_subplot(gs_traj[0, 2])
EMPS_ax_x1 = fig_traj.add_subplot(gs_traj[1, 2])
EMPS_ax_F = fig_traj.add_subplot(gs_traj[2, 2])
fig_traj.set_layout_engine('tight')



### define layout for function plots

# the figure
fig_fcn = plt.figure(dpi=150)
gs_fcn = matplotlib.gridspec.GridSpec(1,3, figure=fig_fcn)

gs_fcn_0 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 3,  width_ratios=(5, 1, 0.2), height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[0,0]
    )

gs_fcn_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 1,  height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[0,1]
    )

gs_fcn_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 1,  height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[0,2]
    )

# generate axes
SMO_ax_S1_tripc = fig_fcn.add_subplot(gs_fcn_0[1, 0])
SMO_ax_S1_histx = fig_fcn.add_subplot(gs_fcn_0[0, 0], sharex=SMO_ax_S1_tripc)
SMO_ax_S1_histy = fig_fcn.add_subplot(gs_fcn_0[1, 1], sharey=SMO_ax_S1_tripc)
SMO_ax_S1_cax = fig_fcn.add_subplot(gs_fcn_0[1, 2])

Veh_ax_S1_plt = fig_fcn.add_subplot(gs_fcn_1[1, 0])
Veh_ax_S1_histx = fig_fcn.add_subplot(gs_fcn_1[0, 0], sharex=Veh_ax_S1_plt)

EMPS_ax_S1_plt = fig_fcn.add_subplot(gs_fcn_2[1, 0])
EMPS_ax_S1_histx = fig_fcn.add_subplot(gs_fcn_2[0, 0], sharex=EMPS_ax_S1_plt)
fig_fcn.set_layout_engine('tight')

SMO_ax_S1_histx.set_title("Single-Mass-Oscillator")
Veh_ax_S1_histx.set_title("Vehicle")
EMPS_ax_S1_histx.set_title("EMPS")



### make trajectory plot

plot_Data(
    Particles=np.concatenate([SMO_offline_Sigma_X, SMO_offline_Sigma_F[...,None]], axis=-1),
    weights=SMO_offline_weights,
    Reference=np.concatenate([SMO_X,SMO_F_sd[...,None]], axis=-1),
    time=SMO_time,
    axes=[SMO_ax_x0, SMO_ax_x1, SMO_ax_F]
)
SMO_ax_x0.set_ylabel(r"$s$ in $\mathrm{m}$")
SMO_ax_x1.set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
SMO_ax_F.set_ylabel(r"$F$ in $\mathrm{N}$")
SMO_ax_F.set_xlabel(r"Time in $\mathrm{s}$")
SMO_ax_x0.set_ylim(-0.8,0.8)
SMO_ax_x1.set_ylim(-3.0,3.0)
SMO_ax_F.set_ylim(-8,8)
SMO_ax_x0.set_title("Single-Mass-Oscillator")
SMO_ax_x0.legend(["mean", f"$3\sigma$", "true"], labelspacing=.07, handlelength=1.0, loc='upper right', fontsize=5)


plot_Data(
    Particles=np.concatenate([Veh_offline_Sigma_X, Veh_offline_Sigma_mu_f[...,None]], axis=-1),
    weights=Veh_offline_weights,
    Reference=np.concatenate([Veh_X, Veh_mu_f[...,None]], axis=-1),
    time=Veh_time,
    axes=[Veh_ax_x0, Veh_ax_x1, Veh_ax_muf]
)
Veh_ax_x0.set_ylabel(r"$\psi$ in $\mathrm{rad/s}$")
Veh_ax_x1.set_ylabel(r"$v_y$ in $\mathrm{m/s}$")
Veh_ax_muf.set_ylabel(r"$\mu_\mathrm{f}$")
Veh_ax_muf.set_xlabel(r"Time in $\mathrm{s}$")
Veh_ax_muf.set_ylim(-1.2,1.2)
Veh_ax_x0.set_title("Vehicle")


plot_Data(
    Particles=np.concatenate([EMPS_offline_Sigma_X, EMPS_offline_Sigma_F[...,None]], axis=-1),
    weights=EMPS_offline_weights,
    Reference=np.concatenate([EMPS_X, np.ones((EMPS_Y.shape[0],1))*np.nan], axis=-1),
    time=EMPS_time,
    axes=[EMPS_ax_x0, EMPS_ax_x1, EMPS_ax_F]
)
EMPS_ax_x0.set_ylabel(r"$q$ in m")
EMPS_ax_x1.set_ylabel(r"$\dot{q}$ in m/s")
EMPS_ax_F.set_ylabel(r"$F$ in N")
EMPS_ax_F.set_xlabel(r"Time in s")
EMPS_ax_x0.set_title("EMPS")


apply_basic_formatting(fig_traj, width=18, height=10, font_size=8)
fig_traj.savefig(r"plots\results_traj_offline.pdf", bbox_inches='tight')



### make function plot

## SMO

# function value from GP
mean, col_scale, row_scale, _ = prior_mniw_Predictive(
    mean=SMO_offline_GP_Mean[-1], 
    col_cov=SMO_offline_GP_Col_Cov[-1], 
    row_scale=SMO_offline_GP_Row_Scale[-1], 
    df=SMO_offline_GP_df[-1], 
    basis=SMO_basis_plot)
SMO_offline_fcn_var = np.diag(col_scale-1) * row_scale[0,0]
SMO_offline_fcn_mean = mean

# calculate wRMSE
SMO_offline_wRMSE = calc_wRMSE(1/SMO_offline_fcn_var, SMO_offline_fcn_mean, SMO_F_sd_true_plot)

# normalize variance to create transparency effect
fcn_alpha = np.maximum(np.minimum(1 - SMO_offline_fcn_var/SMO_fcn_var_prior, 1), 0)

# first slice
plot_fcn_error_2D(
        SMO_X_plot, 
        Mean=np.abs(SMO_offline_fcn_mean-SMO_F_sd_true_plot), 
        X_stats=SMO_offline_Sigma_X, 
        X_weights=SMO_offline_weights, 
        alpha=fcn_alpha,
        fig=fig_fcn,
        ax=SMO_ax_S1_tripc,
        ax_histx=SMO_ax_S1_histx,
        ax_histy=SMO_ax_S1_histy,
        cax=SMO_ax_S1_cax
        )
# SMO_ax_S1_tripc.set_xlabel(r"$s$ in $\mathrm{m}$")
SMO_ax_S1_tripc.set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
SMO_ax_S1_tripc.set_ylim(-3.5, 3.5)
SMO_ax_S1_tripc.set_xlim(-3.5, 3.5)
SMO_ax_S1_tripc.set_xticks([-2,0,2],['$-2$',r'$s$ in $\mathrm{m}$','$2$'])
SMO_ax_S1_histx.set_ylim(0, 100)
SMO_ax_S1_histy.set_xlim(0, 50)
SMO_ax_S1_histx.text(-3.3,46,r'$\# \mathrm{Data}$')



## Vehicle

# function value from GP
mean, col_scale, row_scale, _ = prior_mniw_Predictive(
    mean=Veh_offline_GP_Mean_f[-1], 
    col_cov=Veh_offline_GP_Col_Cov_f[-1], 
    row_scale=Veh_offline_GP_Row_Scale_f[-1], 
    df=Veh_offline_GP_df_f[-1], 
    basis=Veh_basis_plot)
Veh_offline_fcn_var_f = np.diag(col_scale-1) * row_scale[0,0]
Veh_offline_fcn_mean_f = mean
    
mean, col_scale, row_scale, _ = prior_mniw_Predictive(
    mean=Veh_offline_GP_Mean_r[-1], 
    col_cov=Veh_offline_GP_Col_Cov_r[-1], 
    row_scale=Veh_offline_GP_Row_Scale_r[-1], 
    df=Veh_offline_GP_df_r[-1], 
    basis=Veh_basis_plot)
Veh_offline_fcn_var_r = np.diag(col_scale-1) * row_scale[0,0]
Veh_offline_fcn_mean_r = mean

# calculate wRMSE
Veh_offline_wRMSE_f = calc_wRMSE(1/Veh_offline_fcn_var_f, Veh_offline_fcn_mean_f, Veh_mu_true_plot)
Veh_offline_wRMSE_r = calc_wRMSE(1/Veh_offline_fcn_var_r, Veh_offline_fcn_mean_r, Veh_mu_true_plot)

# first slice
plot_fcn_error_1D(
    Veh_alpha_plot, 
    Mean=Veh_offline_fcn_mean_f, 
    Std=np.sqrt(Veh_offline_fcn_var_f),
    X_stats=Veh_offline_Sigma_alpha_f, 
    X_weights=Veh_offline_weights,
    ax=[Veh_ax_S1_plt],
    ax_histx=Veh_ax_S1_histx
    )
Veh_ax_S1_plt.set_xticks([-0.15,0,0.15],['$-0.15$',r'$\alpha$ in $\mathrm{rad}$','$0.15$'])
Veh_ax_S1_plt.set_ylabel(r"$\mu_\mathrm{f}$")
Veh_ax_S1_plt.plot(Veh_alpha_plot, Veh_mu_true_plot, color='red', linestyle=':')
Veh_ax_S1_plt.set_ylim(-1.3, 1.3)
Veh_ax_S1_plt.set_xlim(-0.19, 0.19)
Veh_ax_S1_histx.set_ylim(0, 400)
# Veh_ax_S1_histx.text(-0.31,227,r'$\# \mathrm{Data}$')


## EMPS

# function value from GP
mean, col_scale, row_scale, _ = prior_mniw_Predictive(
    mean=EMPS_offline_GP_Mean[-1], 
    col_cov=EMPS_offline_GP_Col_Cov[-1], 
    row_scale=EMPS_offline_GP_Row_Scale[-1], 
    df=EMPS_offline_GP_df[-1], 
    basis=EMPS_basis_plot)
EMPS_offline_fcn_var_offline = np.diag(col_scale-1) * row_scale[0,0]
EMPS_offline_fcn_mean_offline = mean

# # calculate RMSE over iterations
# RMSE = np.zeros((N_iterations,))
# for i in range(0, N_iterations):
#     RMSE[i] = np.sqrt(np.mean((EMPS_Y - EMPS_offline_Sigma_Y[:,i])**2))

# cumMean_RMSE = np.cumsum(RMSE)/np.arange(1,len(RMSE)+1)

# current_best = RMSE[0]
# for i in range(1, N_iterations):
#     if RMSE[i] < current_best:
#         current_best = RMSE[i]
#     RMSE[i] = current_best

# first slice
plot_fcn_error_1D(
    EMPS_dq_plot, 
    Mean=EMPS_offline_fcn_mean_offline, 
    Std=np.sqrt(EMPS_offline_fcn_var_offline),
    X_stats=EMPS_offline_Sigma_X[...,1], 
    X_weights=EMPS_offline_weights,
    ax=[EMPS_ax_S1_plt],
    ax_histx=EMPS_ax_S1_histx
    )
# EMPS_ax_S1_plt.set_xlabel(r"$\dot{q}$ in m/s")
EMPS_ax_S1_plt.set_ylabel(r"$F$ in N")
EMPS_ax_S1_histx.set_ylim(0, 110)
EMPS_ax_S1_plt.set_xticks([-0.1,0,0.1],['$-0.1$',r'$\dot{q}$ in m/s','$0.1$'])
EMPS_ax_S1_plt.set_ylabel(r"$F$ in N")
EMPS_ax_S1_plt.set_ylim(-58,58)
EMPS_ax_S1_plt.plot(EMPS_dq_plot, EMPS_dq_plot*203.5 + 20.39*np.sign(EMPS_dq_plot) + 3.16, color='r', linestyle=':')


apply_basic_formatting(fig_fcn, width=18, height=4.5, font_size=8)
fig_fcn.savefig(r"plots\results_fcn_offline.pdf", bbox_inches='tight')





################################################################################
# Online

### define layout for trajectory plots

# the figure
fig_traj = plt.figure(dpi=150)
gs_traj = fig_traj.add_gridspec(3, 3,  width_ratios=np.ones(3), height_ratios=np.ones(3))

# generate axes
SMO_ax_x0 = fig_traj.add_subplot(gs_traj[0, 0])
SMO_ax_x1 = fig_traj.add_subplot(gs_traj[1, 0])
SMO_ax_F = fig_traj.add_subplot(gs_traj[2, 0])

Veh_ax_x0 = fig_traj.add_subplot(gs_traj[0, 1])
Veh_ax_x1 = fig_traj.add_subplot(gs_traj[1, 1])
Veh_ax_muf = fig_traj.add_subplot(gs_traj[2, 1])

EMPS_ax_x0 = fig_traj.add_subplot(gs_traj[0, 2])
EMPS_ax_x1 = fig_traj.add_subplot(gs_traj[1, 2])
EMPS_ax_F = fig_traj.add_subplot(gs_traj[2, 2])
fig_traj.set_layout_engine('tight')



### define layout for function plots

# the figure
fig_fcn = plt.figure(dpi=150)
gs_fcn = matplotlib.gridspec.GridSpec(3,3, figure=fig_fcn)

gs_fcn_01 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 3,  width_ratios=(5, 1, 0.2), height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[1,0]
    )
gs_fcn_02 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 3,  width_ratios=(5, 1, 0.2), height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[2,0]
    )

gs_fcn_11 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 1,  height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[1,1]
    )
gs_fcn_12 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 1,  height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[2,1]
    )

gs_fcn_21 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 1,  height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[1,2]
    )
gs_fcn_22 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    2, 1,  height_ratios=(1, 5), 
    hspace=0.05, wspace=0.05, 
    subplot_spec=gs_fcn[2,2]
    )

# generate axes
SMO_ax_wRMSE = fig_fcn.add_subplot(gs_fcn[0, 0])
SMO_ax_S1_tripc = fig_fcn.add_subplot(gs_fcn_01[1, 0])
SMO_ax_S1_histx = fig_fcn.add_subplot(gs_fcn_01[0, 0], sharex=SMO_ax_S1_tripc)
SMO_ax_S1_histy = fig_fcn.add_subplot(gs_fcn_01[1, 1], sharey=SMO_ax_S1_tripc)
SMO_ax_S1_cax = fig_fcn.add_subplot(gs_fcn_01[1, 2])
SMO_ax_S2_tripc = fig_fcn.add_subplot(gs_fcn_02[1, 0])
SMO_ax_S2_histx = fig_fcn.add_subplot(gs_fcn_02[0, 0], sharex=SMO_ax_S2_tripc)
SMO_ax_S2_histy = fig_fcn.add_subplot(gs_fcn_02[1, 1], sharey=SMO_ax_S2_tripc)
SMO_ax_S2_cax = fig_fcn.add_subplot(gs_fcn_02[1, 2])

Veh_ax_wRMSE = fig_fcn.add_subplot(gs_fcn[0, 1])
Veh_ax_S1_plt = fig_fcn.add_subplot(gs_fcn_11[1, 0])
Veh_ax_S1_histx = fig_fcn.add_subplot(gs_fcn_11[0, 0], sharex=Veh_ax_S1_plt)
Veh_ax_S2_plt = fig_fcn.add_subplot(gs_fcn_12[1, 0])
Veh_ax_S2_histx = fig_fcn.add_subplot(gs_fcn_12[0, 0], sharex=Veh_ax_S2_plt)

EMPS_ax_RMSE = fig_fcn.add_subplot(gs_fcn[0, 2])
EMPS_ax_S1_plt = fig_fcn.add_subplot(gs_fcn_21[1, 0])
EMPS_ax_S1_histx = fig_fcn.add_subplot(gs_fcn_21[0, 0], sharex=EMPS_ax_S1_plt)
EMPS_ax_S2_plt = fig_fcn.add_subplot(gs_fcn_22[1, 0])
EMPS_ax_S2_histx = fig_fcn.add_subplot(gs_fcn_22[0, 0], sharex=EMPS_ax_S1_plt)
fig_fcn.set_layout_engine('tight')

SMO_ax_wRMSE.set_title("Single-Mass-Oscillator")
Veh_ax_wRMSE.set_title("Vehicle")
EMPS_ax_RMSE.set_title("EMPS")



### make trajectory plot

plot_Data(
    Particles=np.concatenate([SMO_online_Sigma_X, SMO_online_Sigma_F[...,None]], axis=-1),
    weights=SMO_online_weights,
    Reference=np.concatenate([SMO_X,SMO_F_sd[...,None]], axis=-1),
    time=SMO_time,
    axes=[SMO_ax_x0, SMO_ax_x1, SMO_ax_F]
)
SMO_ax_x0.set_ylabel(r"$s$ in $\mathrm{m}$")
SMO_ax_x1.set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
SMO_ax_F.set_ylabel(r"$F$ in $\mathrm{N}$")
SMO_ax_F.set_xlabel(r"Time in $\mathrm{s}$")
SMO_ax_x0.set_ylim(-0.8,0.8)
SMO_ax_x1.set_ylim(-3.0,3.0)
SMO_ax_F.set_ylim(-8,8)
SMO_ax_x0.set_title("Single-Mass-Oscillator")
SMO_ax_x0.legend(["mean", f"$3\sigma$", "true"], labelspacing=.07, handlelength=1.0, loc='upper right', fontsize=5)


plot_Data(
    Particles=np.concatenate([Veh_online_Sigma_X, Veh_online_Sigma_mu_f[...,None]], axis=-1),
    weights=Veh_online_weights,
    Reference=np.concatenate([Veh_X, Veh_mu_f[...,None]], axis=-1),
    time=Veh_time,
    axes=[Veh_ax_x0, Veh_ax_x1, Veh_ax_muf]
)
Veh_ax_x0.set_ylabel(r"$\psi$ in $\mathrm{rad/s}$")
Veh_ax_x1.set_ylabel(r"$v_y$ in $\mathrm{m/s}$")
Veh_ax_muf.set_ylabel(r"$\mu_\mathrm{f}$")
Veh_ax_muf.set_xlabel(r"Time in $\mathrm{s}$")
Veh_ax_muf.set_ylim(-1.2,1.2)
Veh_ax_x0.set_title("Vehicle")


plot_Data(
    Particles=np.concatenate([EMPS_online_Sigma_X, EMPS_online_Sigma_F[...,None]], axis=-1),
    weights=EMPS_online_weights,
    Reference=np.concatenate([EMPS_X, np.ones((EMPS_Y.shape[0],1))*np.nan], axis=-1),
    time=EMPS_time,
    axes=[EMPS_ax_x0, EMPS_ax_x1, EMPS_ax_F]
)
EMPS_ax_x0.set_ylabel(r"$q$ in m")
EMPS_ax_x1.set_ylabel(r"$\dot{q}$ in m/s")
EMPS_ax_F.set_ylabel(r"$F$ in N")
EMPS_ax_F.set_xlabel(r"Time in s")
EMPS_ax_x0.set_title("EMPS")


apply_basic_formatting(fig_traj, width=18, height=10, font_size=8)
fig_traj.savefig(r"plots\results_traj_online.pdf", bbox_inches='tight')



### make function plot

## SMO

# function value from GP
steps = SMO_time.shape[0]
fcn_mean = np.zeros((steps, SMO_X_plot.shape[0]))
fcn_var = np.zeros((steps, SMO_X_plot.shape[0]))
for i in tqdm(range(0, steps), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=SMO_online_GP_Mean[i], 
        col_cov=SMO_online_GP_Col_Cov[i], 
        row_scale=SMO_online_GP_Row_Scale[i], 
        df=SMO_online_GP_df[i], 
        basis=SMO_basis_plot)
    fcn_var[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean[i] = mean

# normalize variance to create transparency effect
fcn_alpha = np.maximum(np.minimum(1 - fcn_var/SMO_fcn_var_prior, 1), 0)

# calculate wRMSE over iterations
SMO_online_wRMSE = calc_wRMSE(1/fcn_var, fcn_mean, SMO_F_sd_true_plot)

# wRMSE over iterations
SMO_ax_wRMSE.plot(
    SMO_time,
    SMO_online_wRMSE,
    color=imes_blue
    )
SMO_ax_wRMSE.plot([SMO_time[0],SMO_time[-1]], [SMO_offline_wRMSE, SMO_offline_wRMSE],color=imes_blue,linestyle=':')
SMO_ax_wRMSE.set_ylabel("wRMSE in N")
SMO_ax_wRMSE.set_xlim(SMO_time[0], SMO_time[-1])
SMO_ax_wRMSE.set_xticks([0,5,10, SMO_time[-1]],['$0$',r'Time in s','$10$', '$T$'])
SMO_ax_wRMSE.legend(["online", "offline"], labelspacing=.07, handlelength=1.0, loc='upper right', fontsize=5)

# first slice
plot_fcn_error_2D(
        SMO_X_plot, 
        Mean=np.abs(fcn_mean[int(APF_SMO_slice_idx1)]-SMO_F_sd_true_plot), 
        X_stats=SMO_online_Sigma_X[:int(APF_SMO_slice_idx1)], 
        X_weights=SMO_online_weights[:int(APF_SMO_slice_idx1)], 
        alpha=fcn_alpha[int(APF_SMO_slice_idx1)],
        fig=fig_fcn,
        ax=SMO_ax_S1_tripc,
        ax_histx=SMO_ax_S1_histx,
        ax_histy=SMO_ax_S1_histy,
        cax=SMO_ax_S1_cax
        )
SMO_ax_S1_tripc.set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
SMO_ax_S1_tripc.set_ylim(-3.5, 3.5)
SMO_ax_S1_tripc.set_xlim(-3.5, 3.5)
SMO_ax_S1_tripc.set_xticks([-2,0,2],['$-2$',r'$s$ in $\mathrm{m}$','$2$'])
SMO_ax_S1_histx.set_ylim(0, 100)
SMO_ax_S1_histy.set_xlim(0, 50)
SMO_ax_S1_histx.text(-3.3,46,r'$\# \mathrm{Data}$')
# SMO_ax_S1_histx.set_title(f"Time $s={np.round(APF_slice1,1)}*T$")

# second slice
plot_fcn_error_2D(
        SMO_X_plot, 
        Mean=np.abs(fcn_mean[-1]-SMO_F_sd_true_plot), 
        X_stats=SMO_online_Sigma_X[:-1], 
        X_weights=SMO_online_weights[:-1], 
        alpha=fcn_alpha[-1],
        fig=fig_fcn,
        ax=SMO_ax_S2_tripc,
        ax_histx=SMO_ax_S2_histx,
        ax_histy=SMO_ax_S2_histy,
        cax=SMO_ax_S2_cax
        )
SMO_ax_S2_tripc.set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
SMO_ax_S2_tripc.set_ylim(-3.5, 3.5)
SMO_ax_S2_tripc.set_xlim(-3.5, 3.5)
SMO_ax_S2_tripc.set_xticks([-2,0,2],['$-2$',r'$s$ in $\mathrm{m}$','$2$'])
SMO_ax_S2_histx.set_ylim(0, 100)
SMO_ax_S2_histy.set_xlim(0, 50)
# SMO_ax_S2_histx.set_title(f"Time $s=1.0*T$")



## Vehicle

# function value from GP
steps = Veh_time.shape[0]
fcn_mean_f = np.zeros((steps, Veh_alpha_plot.shape[0]))
fcn_var_f = np.zeros((steps, Veh_alpha_plot.shape[0]))
fcn_mean_r = np.zeros((steps, Veh_alpha_plot.shape[0]))
fcn_var_r = np.zeros((steps, Veh_alpha_plot.shape[0]))
for i in tqdm(range(0, steps), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=Veh_online_GP_Mean_f[i], 
        col_cov=Veh_online_GP_Col_Cov_f[i], 
        row_scale=Veh_online_GP_Row_Scale_f[i], 
        df=Veh_online_GP_df_f[i], 
        basis=Veh_basis_plot)
    fcn_var_f[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_f[i] = mean
    
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=Veh_online_GP_Mean_r[i], 
        col_cov=Veh_online_GP_Col_Cov_r[i], 
        row_scale=Veh_online_GP_Row_Scale_r[i], 
        df=Veh_online_GP_df_r[i], 
        basis=Veh_basis_plot)
    fcn_var_r[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_r[i] = mean

# calculate wRMSE over iterations
Veh_online_wRMSE_f = calc_wRMSE(1/fcn_var_f, fcn_mean_f, Veh_mu_true_plot)
Veh_online_wRMSE_r = calc_wRMSE(1/fcn_var_r, fcn_mean_r, Veh_mu_true_plot)

# wRMSE over iterations
Veh_ax_wRMSE.plot(
    Veh_time,
    Veh_online_wRMSE_f,
    color=imes_blue
)
Veh_ax_wRMSE.plot(
    Veh_time,
    Veh_online_wRMSE_r,
    color=imes_orange
)
Veh_ax_wRMSE.plot([Veh_time[0],Veh_time[-1]], [Veh_offline_wRMSE_f, Veh_offline_wRMSE_f],linestyle='--',color=imes_blue)
Veh_ax_wRMSE.plot([Veh_time[0],Veh_time[-1]], [Veh_offline_wRMSE_r, Veh_offline_wRMSE_r],linestyle=':',color=imes_orange)
Veh_ax_wRMSE.legend(["front", "rear"], labelspacing=.07, handlelength=1.0, fontsize=5)
Veh_ax_wRMSE.set_ylabel("wRMSE")
Veh_ax_wRMSE.set_xlim(Veh_time[0], Veh_time[-1])
Veh_ax_wRMSE.set_xticks([0,10,20, Veh_time[-1]],['$0$',r'Time in s','$20$', '$T$'])

# first slice
plot_fcn_error_1D(
    Veh_alpha_plot, 
    Mean=fcn_mean_f[int(APF_Veh_slice_idx1)], 
    Std=np.sqrt(fcn_var_f[int(APF_Veh_slice_idx1)]),
    X_stats=Veh_online_Sigma_alpha_f[:int(APF_Veh_slice_idx1)], 
    X_weights=Veh_online_weights[:int(APF_Veh_slice_idx1)],
    ax=[Veh_ax_S1_plt],
    ax_histx=Veh_ax_S1_histx
    )
Veh_ax_S1_plt.plot(Veh_alpha_plot, Veh_offline_fcn_mean_f, color='gray', linestyle=':')
Veh_ax_S1_plt.set_xticks([-0.15,0,0.15],['$-0.15$',r'$\alpha$ in $\mathrm{rad}$','$0.15$'])
Veh_ax_S1_plt.set_ylabel(r"$\mu_\mathrm{f}$")
Veh_ax_S1_plt.plot(Veh_alpha_plot, Veh_mu_true_plot, color='red', linestyle=':')
Veh_ax_S1_plt.set_ylim(-1.3, 1.3)
Veh_ax_S1_plt.set_xlim(-0.19, 0.19)
Veh_ax_S1_histx.set_ylim(0, 400)
Veh_ax_S1_histx.set_title(f"Time: ${np.round(APF_slice1,1)}\cdot T$")
# Veh_ax_S1_histx.text(-0.31,227,r'$\# \mathrm{Data}$')

# second slice
plot_fcn_error_1D(
    Veh_alpha_plot, 
    Mean=fcn_mean_f[-1], 
    Std=np.sqrt(fcn_var_f[-1]),
    X_stats=Veh_online_Sigma_alpha_f[:-1], 
    X_weights=Veh_online_weights[:-1],
    ax=[Veh_ax_S2_plt],
    ax_histx=Veh_ax_S2_histx
    )
Veh_ax_S2_plt.plot(Veh_alpha_plot, Veh_offline_fcn_mean_f, color='gray', linestyle=':')
Veh_ax_S2_plt.set_xticks([-0.15,0,0.15],['$-0.15$',r'$\alpha$ in $\mathrm{rad}$','$0.15$'])
Veh_ax_S2_plt.set_ylabel(r"$\mu_\mathrm{f}$")
Veh_ax_S2_plt.plot(Veh_alpha_plot, Veh_mu_true_plot, color='red', linestyle=':')
Veh_ax_S2_plt.set_ylim(-1.3, 1.3)
Veh_ax_S2_plt.set_xlim(-0.19, 0.19)
Veh_ax_S2_histx.set_ylim(0, 400)
Veh_ax_S2_histx.set_title(f"Time: $1.0\cdot T$")
Veh_ax_S2_plt.legend(["mean", f"$3\sigma$", "offline", "true"], labelspacing=.07, handlelength=1.0, loc='lower right', fontsize=5)


## EMPS

# function value from GP
steps = EMPS_time.shape[0]
fcn_mean_online = np.zeros((steps, EMPS_dq_plot.shape[0]))
fcn_var_online = np.zeros((steps, EMPS_dq_plot.shape[0]))
for i in tqdm(range(0, steps), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=EMPS_online_GP_Mean[i], 
        col_cov=EMPS_online_GP_Col_Cov[i], 
        row_scale=EMPS_online_GP_Row_Scale[i], 
        df=EMPS_online_GP_df[i], 
        basis=EMPS_basis_plot)
    fcn_var_online[i] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_online[i] = mean

# calculate wRMSE over time
EMPS_online_wRMSE = calc_wRMSE(1/fcn_var_online, fcn_mean_online, EMPS_offline_fcn_mean_offline)

# wRMSE over time
EMPS_ax_RMSE.plot(
    EMPS_time,
    EMPS_online_wRMSE,
    color=imes_blue
)
EMPS_ax_RMSE.set_ylabel("wRMSE in N")
EMPS_ax_RMSE.set_xlim(EMPS_time[0], EMPS_time[-1])
EMPS_ax_RMSE.set_xticks([0,10,20, EMPS_time[-1]],['$0$',r'Time in s','$20$', '$T$'])

# first slice
plot_fcn_error_1D(
    EMPS_dq_plot, 
    Mean=fcn_mean_online[APF_EMPS_slice_idx1], 
    Std=np.sqrt(fcn_var_online[APF_EMPS_slice_idx1]),
    X_stats=EMPS_online_Sigma_X[:APF_EMPS_slice_idx1,:,1], 
    X_weights=EMPS_online_weights[:APF_EMPS_slice_idx1],
    ax=[EMPS_ax_S1_plt],
    ax_histx=EMPS_ax_S1_histx
    )
EMPS_ax_S1_plt.plot(EMPS_dq_plot, EMPS_offline_fcn_mean_offline, color='gray', linestyle=':')
EMPS_ax_S1_plt.set_xticks([-0.1,0,0.1],['$-0.1$',r'$\dot{q}$ in m/s','$0.1$'])
EMPS_ax_S1_plt.set_ylabel(r"$F$ in N")
EMPS_ax_S1_plt.set_ylim(-58,58)
EMPS_ax_S1_histx.set_ylim(0, 60)
# EMPS_ax_S1_histx.set_title(f"Time $s={np.round(APF_slice1,1)}*T$")
# EMPS_ax_S1_histx.text(-0.14,32,r'$\# \mathrm{Data}$')

# second slice
plot_fcn_error_1D(
    EMPS_dq_plot, 
    Mean=fcn_mean_online[-1], 
    Std=np.sqrt(fcn_var_online[-1]),
    X_stats=EMPS_online_Sigma_X[:-1,:,1], 
    X_weights=EMPS_online_weights[:-1],
    ax=[EMPS_ax_S2_plt],
    ax_histx=EMPS_ax_S2_histx
    )
EMPS_ax_S2_plt.plot(EMPS_dq_plot, EMPS_offline_fcn_mean_offline, color='gray', linestyle=':')
EMPS_ax_S2_plt.set_xticks([-0.1,0,0.1],['$-0.1$',r'$\dot{q}$ in m/s','$0.1$'])
EMPS_ax_S2_plt.set_ylabel(r"$F$ in N")
EMPS_ax_S2_plt.set_ylim(-58,58)
EMPS_ax_S2_histx.set_ylim(0, 60)
# EMPS_ax_S2_histx.set_title(f"Time $s=1.0*T$")


apply_basic_formatting(fig_fcn, width=18, height=13, font_size=8)
fig_fcn.savefig(r"plots\results_fcn_online.pdf", bbox_inches='tight')


################################################################################

plt.show()