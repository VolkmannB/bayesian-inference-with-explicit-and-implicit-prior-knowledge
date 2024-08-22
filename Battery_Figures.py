import numpy as np
import jax
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt

from src.Battery import scale_C1, scale_R1
from src.Battery import offset_C1, offset_R1, steps
from src.Publication_Plotting import plot_fcn_error_1D
from src.Publication_Plotting import apply_basic_formatting, plot_Data
from src.BayesianInferrence import prior_mniw_Predictive, prior_mniw_2naturalPara_inv



################################################################################
# Model

N_slices = 2


data = scipy.io.loadmat('Battery.mat')


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



################################################################################
# Plotting Offline

# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=offline_Sigma_Y[...,None],
    weights=offline_weights,
    Reference=Y,
    time=time
)
axes_X[0].set_ylabel(r"$V$ in $\mathrm{V}$")
axes_X[0].set_xlabel(r"Time in $\mathrm{s}$")
apply_basic_formatting(fig_X, width=10, aspect_ratio=0.6, font_size=8)
fig_X.savefig("Battery_PGAS_Y.svg", bbox_inches='tight')

N_PGAS_iter = offline_Sigma_X.shape[1]
index = (np.array(range(N_slices))+1)/N_slices*(N_PGAS_iter-1)

# function value from GP
fcn_mean = np.zeros((N_PGAS_iter, 2, X_plot.shape[0]))
fcn_var = np.zeros((N_PGAS_iter, 2, X_plot.shape[0]))
for i in tqdm(range(0, N_PGAS_iter), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=offline_GP_Mean[i], 
        col_cov=offline_GP_Col_Cov[i], 
        row_scale=offline_GP_Row_Scale[i], 
        df=offline_GP_df[i], 
        basis=basis_plot)
    fcn_var[i,0,:] = np.diag(col_scale-1) * row_scale[0,0] * scale_C1**2
    fcn_var[i,1,:] = np.diag(col_scale-1) * row_scale[1,1] * scale_R1**2
    fcn_mean[i] = ((mean + np.array([offset_C1, offset_R1])) * np.array([scale_C1, scale_R1])).T

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        X_plot, 
        Mean=fcn_mean[int(i)], 
        Std=np.sqrt(fcn_var[int(i)]),
        X_stats=offline_Sigma_X[:,:int(i)], 
        X_weights=offline_weights[:,:int(i)])
    ax_fcn_e[0][1].set_xlabel(r"Voltage in $\mathrm{V}$")
    ax_fcn_e[0][0].set_ylabel(r"$C_1$ in $\mathrm{F}$")
    ax_fcn_e[0][1].set_ylabel(r"$R_1$ in $\mathrm{\Omega}$")
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"Battery_PGAS_C1R1_fcn_{int(i)}.svg")



plt.show()