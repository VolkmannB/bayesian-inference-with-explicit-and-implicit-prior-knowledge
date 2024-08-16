import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import pandas as pd

import matplotlib.pyplot as plt

from src.Battery import Battery_CPFAS_Kernel, basis_fcn, scale_C1, scale_R1
from src.Battery import steps, N_basis_fcn, GP_prior_C1R1, offset_C1, offset_R1
from src.Battery import  forget_factor
from src.Publication_Plotting import apply_basic_formatting, plot_fcn_error_1D
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_calcStatistics
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_Predictive



################################################################################
### Offline Algorithm

N_iterations = 2
Sigma_X = np.zeros((steps,N_iterations))
Sigma_C1R1 = np.zeros((steps,N_iterations,2))
weights = np.ones((steps,N_iterations))/N_iterations

# logging of model
Mean_C1R1 = np.zeros((N_iterations, 2, N_basis_fcn)) # GP
Col_Cov_C1R1 = np.zeros((N_iterations, N_basis_fcn, N_basis_fcn)) # GP
Row_Scale_C1R1 = np.zeros((N_iterations, 2, 2)) # GP
df_C1R1 = np.zeros((N_iterations,)) # GP

# variable for sufficient statistics
GP_model_stats = [
    np.zeros((N_basis_fcn, 2)),
    np.zeros((N_basis_fcn, N_basis_fcn)),
    np.zeros((2, 2)),
    0
]
GP_model_stats_logging = [
    np.zeros((N_basis_fcn, 2)),
    np.zeros((N_basis_fcn, N_basis_fcn)),
    np.zeros((2, 2)),
    0
]

# set initial reference using APF
print(f"\nSetting initial reference trajectory")
GP_para = prior_mniw_2naturalPara_inv(
    GP_prior_C1R1[0],
    GP_prior_C1R1[1],
    GP_prior_C1R1[2],
    GP_prior_C1R1[3]
)
Sigma_X[:,0], Sigma_C1R1[:,0] = Battery_CPFAS_Kernel(
        x_ref=None,
        C1R1_ref=None,
        Mean_C1R1=GP_para[0], 
        Col_Cov_C1R1=GP_para[1], 
        Row_Scale_C1R1=GP_para[2], 
        df_C1R1=GP_para[3])
    
    
# make proposal for distribution of F_sd using new proposals of trajectories
phi = jax.vmap(basis_fcn)(Sigma_X[:,0])
T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
        Sigma_C1R1[:,0],
        phi
    )
GP_model_stats[0] = np.sum(T_0, axis=0)
GP_model_stats[1] = np.sum(T_1, axis=0)
GP_model_stats[2] = np.sum(T_2, axis=0)
GP_model_stats[3] = np.sum(T_3, axis=0)

# initial model
GP_para = prior_mniw_2naturalPara_inv(
    GP_prior_C1R1[0] + GP_model_stats[0],
    GP_prior_C1R1[1] + GP_model_stats[1],
    GP_prior_C1R1[2] + GP_model_stats[2],
    GP_prior_C1R1[3] + GP_model_stats[3]
)
Mean_C1R1[0] = GP_para[0]
Col_Cov_C1R1[0] = GP_para[1]
Row_Scale_C1R1[0] = GP_para[2]
df_C1R1[0] = GP_para[3]



### Run PGAS
for k in range(1, N_iterations):
    print(f"\nStarting iteration {k}")
        
    # calculate parameters of GP from prior and sufficient statistics
    GP_para = list(prior_mniw_2naturalPara_inv(
        GP_prior_C1R1[0] + GP_model_stats[0],
        GP_prior_C1R1[1] + GP_model_stats[1],
        GP_prior_C1R1[2] + GP_model_stats[2],
        GP_prior_C1R1[3] + GP_model_stats[3]
    ))
    print(f"    Var is {GP_para[2].flatten()/(GP_para[3]+1)}")
    
    
    
    # sample new proposal for trajectories using CPF with AS
    Sigma_X[:,k], Sigma_C1R1[:,k] = Battery_CPFAS_Kernel(
        x_ref=Sigma_X[:,k-1],
        C1R1_ref=Sigma_C1R1[:,k-1],
        Mean_C1R1=GP_para[0], 
        Col_Cov_C1R1=GP_para[1], 
        Row_Scale_C1R1=GP_para[2], 
        df_C1R1=GP_para[3])
    
    
    # make proposal for distribution of F_sd using new proposals of trajectories
    phi = jax.vmap(basis_fcn)(Sigma_X[:,k])
    T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_C1R1[:,k],
            phi
        )
    GP_model_stats[0] = np.sum(T_0, axis=0)
    GP_model_stats[1] = np.sum(T_1, axis=0)
    GP_model_stats[2] = np.sum(T_2, axis=0)
    GP_model_stats[3] = np.sum(T_3, axis=0)
    
    
    # logging
    GP_para = prior_mniw_2naturalPara_inv(
        GP_prior_C1R1[0] + GP_model_stats[0],
        GP_prior_C1R1[1] + GP_model_stats[1],
        GP_prior_C1R1[2] + GP_model_stats[2],
        GP_prior_C1R1[3] + GP_model_stats[3]
    )
    Mean_C1R1[k] = GP_para[0]
    Col_Cov_C1R1[k] = GP_para[1]
    Row_Scale_C1R1[k] = GP_para[2]
    df_C1R1[k] = GP_para[3]
    
    GP_model_stats_logging[0] = GP_model_stats_logging[0] * forget_factor + np.sum(T_0, axis=0)
    GP_model_stats_logging[1] = GP_model_stats_logging[0] * forget_factor + np.sum(T_1, axis=0)
    GP_model_stats_logging[2] = GP_model_stats_logging[0] * forget_factor + np.sum(T_2, axis=0)
    GP_model_stats_logging[3] = GP_model_stats_logging[0] * forget_factor + np.sum(T_3, axis=0)



################################################################################
# Plotting


# plot learned function
V = jnp.linspace(0, 2.2, 500)
basis_in = jax.vmap(basis_fcn)(V)

N_slices = 1
index = (np.array(range(N_slices))+1)/N_slices*(N_iterations-1)

# function value from GP
fcn_mean = np.zeros((N_iterations, 2, V.shape[0]))
fcn_var = np.zeros((N_iterations, 2, V.shape[0]))
for i in tqdm(range(0, N_iterations), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=Mean_C1R1[i], 
        col_cov=Col_Cov_C1R1[i], 
        row_scale=Row_Scale_C1R1[i], 
        df=df_C1R1[i], 
        basis=basis_in)
    fcn_var[i,0,:] = np.diag(col_scale-1) * row_scale[0,0] * scale_C1**2
    fcn_var[i,1,:] = np.diag(col_scale-1) * row_scale[1,1] * scale_R1**2
    fcn_mean[i] = ((mean + np.array([offset_C1, offset_R1])) * np.array([scale_C1, scale_R1])).T

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        V, 
        Mean=fcn_mean[int(i)], 
        Std=np.sqrt(fcn_var[int(i)]),
        X_stats=Sigma_X[:int(i)], 
        X_weights=weights[:int(i)])
    ax_fcn_e[0][1].set_xlabel(r"Voltage in $\mathrm{V}$")
    ax_fcn_e[0][0].set_ylabel(r"$C_1$ in $\mathrm{F}$")
    ax_fcn_e[0][1].set_ylabel(r"$R_1$ in $\mathrm{\Omega}$")
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"Battery_PGAS_C1R1_fcn_{int(i)}.svg")



plt.show()