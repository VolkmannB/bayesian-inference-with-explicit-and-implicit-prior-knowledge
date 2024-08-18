import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools

import matplotlib.pyplot as plt



from src.Vehicle import Vehicle_simulation, Vehicle_APF, basis_fcn, mu_y
from src.Vehicle import time, steps, GP_prior_f
from src.Publication_Plotting import plot_Data, apply_basic_formatting
from src.Publication_Plotting import plot_fcn_error_1D, imes_blue, imes_orange
from src.BayesianInferrence import prior_mniw_Predictive, prior_mniw_2naturalPara_inv



################################################################################

# Simulation
X, Y, Mu_true = Vehicle_simulation()

# Online Algorithm
(
    Sigma_X, 
    Sigma_mu_f, 
    Sigma_mu_r, 
    Sigma_alpha_f, 
    Sigma_alpha_r, 
    weights, 
    Mean_f, 
    Col_Cov_f, 
    Row_Scale_f, 
    df_f, 
    Mean_r, 
    Col_Cov_r, 
    Row_Scale_r, 
    df_r
    ) = Vehicle_APF(Y)




################################################################################
# Plotting


# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=np.concatenate([Sigma_X, Sigma_mu_f[...,None]], axis=-1),
    weights=weights,
    Reference=np.concatenate([X,Mu_true[...,0,None]], axis=-1),
    time=time
)
axes_X[0].set_ylabel(r"$\dot{\psi}$ in $rad/s$")
axes_X[1].set_ylabel(r"$v_y$ in $m/s$")
axes_X[1].set_xlabel(r"Time in $s$")
apply_basic_formatting(fig_X, width=8, aspect_ratio=1,  font_size=8)
fig_X.savefig("VehicleSimulation_APF_X.svg", bbox_inches='tight')

    
    
# plot time slices of the learned MTF for the front tire
alpha = jnp.linspace(-20/180*jnp.pi, 20/180*jnp.pi, 500)
mu_f_true = jax.vmap(functools.partial(mu_y))(alpha=alpha)
mu_r_true = jax.vmap(functools.partial(mu_y))(alpha=alpha)
basis_in = jax.vmap(basis_fcn)(alpha)

N_slices = 4
index2 = (np.array(range(N_slices))+1)/N_slices*(steps-1)
index = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])*(steps-1)
# function values with GP prior
GP_prior = prior_mniw_2naturalPara_inv(
            GP_prior_f[0],
            GP_prior_f[1],
            GP_prior_f[2],
            GP_prior_f[3]
        )
_, col_scale_prior, row_scale_prior, _ = prior_mniw_Predictive(
    mean=GP_prior[0], 
    col_cov=GP_prior[1], 
    row_scale=GP_prior[2], 
    df=GP_prior[3], 
    basis=basis_in)
fcn_var_prior = np.diag(col_scale_prior-1) * row_scale_prior[0,0]
del col_scale_prior, row_scale_prior, GP_prior

# function value from GP front
fcn_mean_f = np.zeros((time.shape[0], alpha.shape[0]))
fcn_var_f = np.zeros((time.shape[0], alpha.shape[0]))
for i in tqdm(range(0, time.shape[0]), desc='Calculating fcn error and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=Mean_f[i], 
        col_cov=Col_Cov_f[i], 
        row_scale=Row_Scale_f[i], 
        df=df_f[i], 
        basis=basis_in)
    fcn_var_f[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_f[i] = mean

# function value from GP rear
fcn_mean_r = np.zeros((time.shape[0], alpha.shape[0]))
fcn_var_r = np.zeros((time.shape[0], alpha.shape[0]))
for i in tqdm(range(0, time.shape[0]), desc='Calculating fcn error and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=Mean_r[i], 
        col_cov=Col_Cov_r[i], 
        row_scale=Row_Scale_r[i], 
        df=df_r[i], 
        basis=basis_in)
    fcn_var_r[i,:] = np.diag(col_scale-1) * row_scale[0,0]
    fcn_mean_r[i] = mean


# normalize variance to create transparency effect
fcn_alpha = np.maximum(np.minimum(1 - fcn_var_f/fcn_var_prior, 1), 0)

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        alpha, 
        Mean=fcn_mean_f[int(i)], 
        Std=np.sqrt(fcn_var_f[int(i)]),
        X_stats=Sigma_mu_f[:int(i)], 
        X_weights=weights[:int(i)],
        max_hist = 80)
    ax_fcn_e[0][-1].set_xlabel(r"$\alpha$ in $\mathrm{rad}$")
    ax_fcn_e[0][-1].plot(alpha, mu_f_true, color='red', linestyle=':')
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"Vehicle_APF_muf_fcn_{np.round(time[int(i)],3)}.svg")



# plot weighted RMSE of GP over entire function space
fcn_var_f = fcn_var_f + 1e-4 # to avoid dividing by zero
v1 = np.sum(1/fcn_var_f, axis=-1)
wRMSE_f = np.sqrt(v1/(v1**2 - v1) * jnp.sum((fcn_mean_f - mu_f_true) ** 2 / fcn_var_f, axis=-1))

fcn_var_r = fcn_var_r + 1e-4 # to avoid dividing by zero
v1 = np.sum(1/fcn_var_r, axis=-1)
wRMSE_r = np.sqrt(v1/(v1**2 - v1) * jnp.sum((fcn_mean_r - mu_r_true) ** 2 / fcn_var_r, axis=-1))

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
    
    
apply_basic_formatting(fig_RMSE, width=8, aspect_ratio=1, font_size=8)
fig_RMSE.savefig("Vehicle_APF_muf_wRMSE.svg", bbox_inches='tight')


################################################################################
# Saving

np.savez('VehicleSimulation_APF_saved.npz',
    X=X, 
    Y=Y, 
    Mu_true=Mu_true,
    Sigma_X=Sigma_X, 
    Sigma_mu_f=Sigma_mu_f, 
    Sigma_mu_r=Sigma_mu_r, 
    Sigma_alpha_f=Sigma_alpha_f, 
    Sigma_alpha_r=Sigma_alpha_r, 
    weights=weights, 
    Mean_f=Mean_f, 
    Col_Cov_f=Col_Cov_f, 
    Row_Scale_f=Row_Scale_f, 
    df_f=df_f, 
    Mean_r=Mean_r, 
    Col_Cov_r=Col_Cov_r, 
    Row_Scale_r=Row_Scale_r, 
    df_r=df_r,
    fcn_var_f=fcn_var_f,
    fcn_mean_f=fcn_mean_f,
    fcn_var_r=fcn_var_r,
    fcn_mean_r=fcn_mean_r,
    alpha=alpha,
    time=time,
    steps=steps,
    )


###
plt.show()
