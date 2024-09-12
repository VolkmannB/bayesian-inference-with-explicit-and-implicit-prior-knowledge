import numpy as np
import jax
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt

from src.Battery import scale_C1, scale_R1
from src.Battery import offset_C1, offset_R1, steps
from src.Publication_Plotting import plot_fcn_error_1D, imes_blue, imes_orange
from src.Publication_Plotting import apply_basic_formatting, plot_Data
from src.BayesianInferrence import prior_mniw_Predictive, prior_mniw_2naturalPara_inv



################################################################################
# Model

N_slices = 2


data = scipy.io.loadmat('plots\Battery.mat')


offline_Sigma_X = data['offline_Sigma_X']
offline_Sigma_C1R1 = data['offline_Sigma_C1R1']
offline_Sigma_Y = data['offline_Sigma_Y']
offline_weights = data['offline_weights']
offline_T0 = data['offline_T0']
offline_T1 = data['offline_T1']
offline_T2 = data['offline_T2']
offline_T3 = data['offline_T3'].flatten()

online_Sigma_X = data['online_Sigma_X']
online_Sigma_C1R1 = data['online_Sigma_C1R1']
online_Sigma_Y = data['online_Sigma_Y']
online_weights = data['online_weights']
online_T0 = data['online_T0']
online_T1 = data['online_T1']
online_T2 = data['online_T2']
online_T3 = data['online_T3'].flatten()

GP_prior_stats = [data['prior_T0'], data['prior_T1'], 
                  data['prior_T2'], data['prior_T3'].flatten()]
scale_C1 = data['scale_C1'].flatten()[0]
scale_R1 = data['scale_R1'].flatten()[0]
offset_C1 = data['offset_C1'].flatten()[0]
offset_R1 = data['offset_R1'].flatten()[0]

X_plot = data['X_plot'].flatten()
basis_plot = data['basis_plot']
time = data['time'].flatten()

Y = data['Y'].flatten()

del data
    
    
    
# Convert sufficient statistics to standard parameters
(offline_GP_Mean, offline_GP_Col_Cov, 
 offline_GP_Row_Scale, offline_GP_df) = jax.vmap(prior_mniw_2naturalPara_inv)(
    GP_prior_stats[0] + np.cumsum(offline_T0, axis=0),
    GP_prior_stats[1] + np.cumsum(offline_T1, axis=0),
    GP_prior_stats[2] + np.cumsum(offline_T2, axis=0),
    GP_prior_stats[3] + np.cumsum(offline_T3, axis=0)
)
del offline_T0, offline_T1, offline_T2, offline_T3
    
# Convert sufficient statistics to standard parameters
(online_GP_Mean, online_GP_Col_Cov, 
 online_GP_Row_Scale, online_GP_df) = jax.vmap(prior_mniw_2naturalPara_inv)(
    GP_prior_stats[0] + online_T0,
    GP_prior_stats[1] + online_T1,
    GP_prior_stats[2] + online_T2,
    GP_prior_stats[3] + online_T3
)
del online_T0, online_T1, online_T2, online_T3



################################################################################
# Plotting Offline

# plot the state estimations
offline_C1R1 = (offline_Sigma_C1R1 + np.array([offset_C1, offset_R1]) )* np.array([scale_C1, scale_R1])
fig_X, axes_X = plot_Data(
    Particles=np.concatenate([offline_Sigma_Y[...,None], offline_C1R1], axis=-1),
    weights=offline_weights,
    Reference=np.concatenate([Y[...,None], np.ones((Y.shape[0],2))*np.nan], axis=-1),
    time=time
)
axes_X[0].set_ylabel(r"$V$ in $\mathrm{V}$")
axes_X[1].set_ylabel(r"$C_1$ in F")
axes_X[2].set_ylabel(r"$R_1$ in $\Omega$")
axes_X[2].set_xlabel(r"Time in $\mathrm{s}$")
apply_basic_formatting(fig_X, width=10, aspect_ratio=0.6, font_size=8)
fig_X.savefig("plots\Battery_PGAS_Y.svg", bbox_inches='tight')

N_PGAS_iter = offline_Sigma_X.shape[1]
index = (np.array(range(N_slices))+1)/N_slices*(N_PGAS_iter-1)

# function value from GP
fcn_mean_offline = np.zeros((N_PGAS_iter, 2, X_plot.shape[0]))
fcn_var_offline = np.zeros((N_PGAS_iter, 2, X_plot.shape[0]))
for i in tqdm(range(0, N_PGAS_iter), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=offline_GP_Mean[i], 
        col_cov=offline_GP_Col_Cov[i], 
        row_scale=offline_GP_Row_Scale[i], 
        df=offline_GP_df[i], 
        basis=basis_plot)
    fcn_var_offline[i,0,:] = np.diag(col_scale-1) * row_scale[0,0] * scale_C1**2
    fcn_var_offline[i,1,:] = np.diag(col_scale-1) * row_scale[1,1] * scale_R1**2
    fcn_mean_offline[i] = ((mean + np.array([offset_C1, offset_R1])) * np.array([scale_C1, scale_R1])).T

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        X_plot, 
        Mean=fcn_mean_offline[int(i)], 
        Std=np.sqrt(fcn_var_offline[int(i)]),
        X_stats=offline_Sigma_X[:,:int(i)], 
        X_weights=offline_weights[:,:int(i)])
    ax_fcn_e[0][1].set_xlabel(r"Voltage in $\mathrm{V}$")
    ax_fcn_e[0][0].set_ylabel(r"$C_1$ in $\mathrm{F}$")
    ax_fcn_e[0][1].set_ylabel(r"$R_1$ in $\mathrm{\Omega}$")
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"plots\Battery_PGAS_C1R1_fcn_{int(i)}.svg")



# plot RMSE between observations and predictions
RMSE = np.zeros((N_PGAS_iter,))
for i in range(0, N_PGAS_iter):
    RMSE[i] = np.sqrt(np.mean((Y - offline_Sigma_Y[:,i])**2))
fig_RMSE, ax_RMSE = plt.subplots(1,1, layout='tight')
ax_RMSE.plot(
    range(0,N_PGAS_iter),
    RMSE,
    color=imes_blue
)
apply_basic_formatting(fig_RMSE, width=8, font_size=8)



################################################################################
# Plotting Online

# plot the state estimations
online_C1R1 = (online_Sigma_C1R1 + np.array([offset_C1, offset_R1]) )* np.array([scale_C1, scale_R1])
fig_X, axes_X = plot_Data(
    Particles=np.concatenate([online_Sigma_Y[...,None], online_C1R1], axis=-1),
    weights=online_weights,
    Reference=np.concatenate([Y[...,None], np.ones((Y.shape[0],2))*np.nan], axis=-1),
    time=time
)
axes_X[0].set_ylabel(r"$V$ in $\mathrm{V}$")
axes_X[1].set_ylabel(r"$C_1$ in F")
axes_X[2].set_ylabel(r"$R_1$ in $\Omega$")
axes_X[2].set_xlabel(r"Time in $\mathrm{s}$")
apply_basic_formatting(fig_X, width=10, aspect_ratio=0.6, font_size=8)
fig_X.savefig("plots\Battery_APF_Y.svg", bbox_inches='tight')

steps = time.shape[0]
index = (np.array(range(N_slices))+1)/N_slices*(steps-1)

# function value from GP
fcn_mean_online = np.zeros((steps, 2, X_plot.shape[0]))
fcn_var_online = np.zeros((steps, 2, X_plot.shape[0]))
for i in tqdm(range(0, steps), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=online_GP_Mean[i], 
        col_cov=online_GP_Col_Cov[i], 
        row_scale=online_GP_Row_Scale[i], 
        df=online_GP_df[i], 
        basis=basis_plot)
    fcn_var_online[i,0,:] = np.diag(col_scale-1) * row_scale[0,0] * scale_C1**2
    fcn_var_online[i,1,:] = np.diag(col_scale-1) * row_scale[1,1] * scale_R1**2
    fcn_mean_online[i] = ((mean + np.array([offset_C1, offset_R1])) * np.array([scale_C1, scale_R1])).T

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        X_plot, 
        Mean=fcn_mean_online[int(i)], 
        Std=np.sqrt(fcn_var_online[int(i)]),
        X_stats=online_Sigma_X[:int(i)], 
        X_weights=online_weights[:int(i)])
    ax_fcn_e[0][1].set_xlabel(r"Voltage in $\mathrm{V}$")
    ax_fcn_e[0][0].set_ylabel(r"$C_1$ in $\mathrm{F}$")
    ax_fcn_e[0][1].set_ylabel(r"$R_1$ in $\mathrm{\Omega}$")
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"plots\Battery_APF_C1R1_fcn_{int(i)}.svg")
    


# plot weighted RMSE of GP over entire function space
fcn_var = fcn_var_online + fcn_var_offline[-1]
w = 1 / fcn_var[:,0,:]
w = w / np.sum(w, axis=-1, keepdims=True)
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE_f = np.sqrt(1/(v1-(v2/v1**2)) * np.sum((fcn_mean_online[:,0] - fcn_mean_offline[-1,0])**2 * w, axis=-1))

w = 1 / fcn_var[:,1,:]
w = w / np.sum(w, axis=-1, keepdims=True)
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE_r = np.sqrt(1/(v1-(v2/v1**2)) * np.sum((fcn_mean_online[:,1] - fcn_mean_offline[-1,1])**2 * w, axis=-1))

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

for i in index:
    ax_RMSE.plot([time[int(i)], time[int(i)]], [0, wRMSE_f[int(i)]*1.5], color="black", linewidth=0.8)
    
    
apply_basic_formatting(fig_RMSE, width=8, font_size=8)
fig_RMSE.savefig("plots\Vehicle_APF_C1R1_wRMSE.svg", bbox_inches='tight')



plt.show()