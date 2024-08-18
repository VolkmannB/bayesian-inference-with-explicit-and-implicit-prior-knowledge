import numpy as np
import jax
import jax.numpy as jnp
import functools
from tqdm import tqdm



import matplotlib.pyplot as plt



from src.SingleMassOscillator import F_spring, F_damper, basis_fcn, time, steps
from src.SingleMassOscillator import forget_factor, dt, GP_model_prior
from src.SingleMassOscillator import SingleMassOscillator_simulation
from src.SingleMassOscillator import SingleMassOscillator_APF
from src.Publication_Plotting import plot_BFE_2D, generate_BFE_TimeSlices, plot_fcn_error_2D
from src.Publication_Plotting import plot_Data, apply_basic_formatting, imes_blue
from src.BayesianInferrence import prior_mniw_Predictive, prior_mniw_2naturalPara_inv



################################################################################

# Simulation
X, Y, F_sd = SingleMassOscillator_simulation()

# Online Algorithm    
Sigma_X, Sigma_F, weights, Mean_F, Col_cov_F, Row_scale_F, df_F = SingleMassOscillator_APF(Y)
    



################################################################################
# Plots


# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=np.concatenate([Sigma_X, Sigma_F[...,None]], axis=-1),
    weights=weights,
    Reference=np.concatenate([X,F_sd[...,None]], axis=-1),
    time=time
)
axes_X[0].set_ylabel(r"$s$ in $\mathrm{m}$")
axes_X[1].set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
axes_X[2].set_ylabel(r"$F$ in $\mathrm{N}$")
axes_X[2].set_xlabel(r"Time in $\mathrm{s}$")
apply_basic_formatting(fig_X, width=8, aspect_ratio=1, font_size=8)
fig_X.savefig("SingleMassOscillator_APF_X.svg", bbox_inches='tight')



### plot time slices of the learned spring-damper function
x_plt = np.linspace(-5., 5., 50)
dx_plt = np.linspace(-5., 5., 50)
grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')
X_in = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
basis_in = jax.vmap(basis_fcn)(X_in)

N_slices = 4
index2 = (np.array(range(N_slices))+1)/N_slices*(steps-1)
index = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])*(steps-1)

# true spring damper force
F_sd_true = jax.vmap(F_spring)(X_in[:,0]) + jax.vmap(F_damper)(X_in[:,1])

# function values with GP prior
GP_prior = prior_mniw_2naturalPara_inv(
            GP_model_prior[0],
            GP_model_prior[1],
            GP_model_prior[2],
            GP_model_prior[3]
        )
_, col_scale_prior, row_scale_prior, _ = prior_mniw_Predictive(
    mean=GP_prior[0], 
    col_cov=GP_prior[1], 
    row_scale=GP_prior[2], 
    df=GP_prior[3], 
    basis=basis_in)
fcn_var_prior = np.diag(col_scale_prior-1) * row_scale_prior[0,0]
del col_scale_prior, row_scale_prior, GP_prior

# function value from GP
fcn_mean = np.zeros((time.shape[0], X_in.shape[0]))
fcn_var = np.zeros((time.shape[0], X_in.shape[0]))
for i in tqdm(range(0, time.shape[0]), desc='Calculating fcn error and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=Mean_F[i], 
        col_cov=Col_cov_F[i], 
        row_scale=Row_scale_F[i], 
        df=df_F[i], 
        basis=basis_in)
    fcn_var[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean[i] = mean

# normalize variance to create transparency effect
fcn_alpha = np.maximum(np.minimum(1 - fcn_var/fcn_var_prior, 1), 0)

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_2D(
        X_in, 
        Mean=np.abs(fcn_mean[int(i)]-F_sd_true), 
        X_stats=Sigma_X[:int(i)], 
        X_weights=weights[:int(i)], 
        alpha=fcn_alpha[int(i)],
        max_x=250, max_y=100)
    ax_fcn_e[0].set_xlabel(r"$s$ in $\mathrm{m}$")
    ax_fcn_e[0].set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"SingleMassOscillator_APF_Fsd_fcn_{np.round(time[int(i)],3)}.svg")




# plot weighted RMSE of GP over entire function space
fcn_var = fcn_var + 1e-4 # to avoid dividing by zero
v1 = np.sum(1/fcn_var, axis=-1)
wRMSE = np.sqrt(v1/(v1**2 - v1) * jnp.sum((fcn_mean - F_sd_true) ** 2 / fcn_var, axis=-1))
fig_RMSE, ax_RMSE = plt.subplots(1,1, layout='tight')
ax_RMSE.plot(
    time,
    wRMSE,
    color=imes_blue
)
ax_RMSE.set_ylabel(r"wRMSE")
ax_RMSE.set_xlabel(r"Time in $\mathrm{s}$")
ax_RMSE.set_ylim(0)

for i in index:
    ax_RMSE.plot([time[int(i)], time[int(i)]], [0, wRMSE[int(i)]*1.5], color="black", linewidth=0.8)
    
apply_basic_formatting(fig_RMSE, width=8, aspect_ratio=1,  font_size=8)
fig_RMSE.savefig("SingleMassOscillator_APF_Fsd_wRMSE.svg", bbox_inches='tight')


################################################################################
# Saving

np.savez('SingleMassOscillator_APF_saved.npz',
    X=X, 
    Y=Y, 
    F_sd=F_sd,
    Sigma_X=Sigma_X, 
    Sigma_F=Sigma_F, 
    weights=weights, 
    Mean_F=Mean_F, 
    Col_cov_F=Col_cov_F, 
    Row_scale_F=Row_scale_F, 
    df_F=df_F,
    fcn_var=fcn_var,
    fcn_mean=fcn_mean,
    time=time,
    steps=steps,
    )


###
plt.show()