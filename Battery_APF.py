import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import pandas as pd

import matplotlib.pyplot as plt

from src.Battery import Battery_APF, basis_fcn, data, time, scale_C1, scale_R1
from src.Battery import offset_C1, offset_R1, GP_prior_C1R1, steps
from src.Publication_Plotting import plot_fcn_error_1D
from src.Publication_Plotting import apply_basic_formatting
from src.BayesianInferrence import prior_mniw_Predictive, prior_mniw_2naturalPara_inv



################################################################################
# Model


Sigma_X, Sigma_C1R1, Sigma_Y, weights, Mean_C1R1, Col_Cov_C1R1, Row_Scale_C1R1, df_C1R1 = Battery_APF()



################################################################################
# Plotting


# plot learned function
V = jnp.linspace(0, 2.2, 500)
basis_in = jax.vmap(basis_fcn)(V)

N_slices = 4
index = (np.array(range(N_slices))+1)/N_slices*(steps-1)

# function value from GP
fcn_mean = np.zeros((steps, 2, V.shape[0]))
fcn_var = np.zeros((steps, 2, V.shape[0]))
for i in tqdm(range(0, steps), desc='Calculating fcn value and var'):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=Mean_C1R1[i], 
        col_cov=Col_Cov_C1R1[i], 
        row_scale=Row_Scale_C1R1[i], 
        df=df_C1R1[i], 
        basis=basis_in)
    fcn_var[i,0,:] = np.diag(col_scale-1) * row_scale[0,0] * scale_C1**2
    fcn_var[i,1,:] = np.diag(col_scale-1) * row_scale[1,1] * scale_R1**2
    fcn_mean[i] = (mean * np.array([scale_C1, scale_R1]) + np.array([offset_C1, offset_R1])).T

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        V, 
        Mean=fcn_mean[int(i)], 
        Std=np.sqrt(fcn_var[int(i)]),
        X_stats=Sigma_X[:int(i)], 
        X_weights=weights[:int(i)])
    ax_fcn_e[0][2].set_xlabel(r"Voltage in $\mathrm{V}$")
    ax_fcn_e[0][1].set_ylabel(r"$C_1$ in $\mathrm{F}$")
    ax_fcn_e[0][2].set_ylabel(r"$R_1$ in $\mathrm{\Omega}$")
        
    apply_basic_formatting(fig_fcn_e, width=8, aspect_ratio=1, font_size=8)
    fig_fcn_e.savefig(f"Battery_APF_C1R1_fcn_{int(i)}.svg")



plt.show()