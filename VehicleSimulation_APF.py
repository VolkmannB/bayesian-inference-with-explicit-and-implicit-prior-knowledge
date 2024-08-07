import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import subprocess



import matplotlib.pyplot as plt



from src.Vehicle import Vehicle_simulation, Vehicle_APF, basis_fcn, mu_y
from src.Vehicle import time, forget_factor, dt, f_alpha
from src.Publication_Plotting import plot_BFE_1D, generate_BFE_TimeSlices
from src.Publication_Plotting import plot_Data, apply_basic_formatting



################################################################################

# Simulation
X, Y = Vehicle_simulation()

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
    df_f, Mean_r, 
    Col_Cov_r, 
    Row_Scale_r, 
    df_r
    ) = Vehicle_APF(Y)


################################################################################
# Plotting


# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=Sigma_X,
    weights=weights,
    Reference=X,
    time=time
)
axes_X[0].set_ylabel(r"$\dot{\psi}$ in $rad/s$")
axes_X[1].set_ylabel(r"$v_y$ in $m/s$")
axes_X[1].set_xlabel(r"Time in $s$")
apply_basic_formatting(fig_X, width=8, font_size=8)
fig_X.savefig("VehicleSimulation_APF_X.pdf", bbox_inches='tight')

    
    
# plot time slices of the learned MTF for the front tire
alpha = jnp.linspace(-20/180*jnp.pi, 20/180*jnp.pi, 500)
mu_f_true = jax.vmap(functools.partial(mu_y))(alpha=alpha)

Mean, Std, X_stats, X_weights, Time = generate_BFE_TimeSlices(
    N_slices=4, 
    X_in=alpha[...,None], 
    Sigma_X=Sigma_alpha_f, 
    Sigma_weights=weights,
    Mean=Mean_f, 
    Col_Scale=Col_Cov_f, 
    Row_Scale=Row_Scale_f, 
    DF=df_f, 
    basis_fcn=basis_fcn, 
    forget_factor=forget_factor
    )

fig_BFE_f, ax_BFE_f = plot_BFE_1D(
    alpha,
    Mean,
    Std,
    Time*dt,
    X_stats, 
    X_weights
)

# plot true function
for ax in ax_BFE_f[0]:
    ax.plot(alpha, mu_f_true, color='red', linestyle='--')

# set x label
for ax in ax_BFE_f[1]:
    ax.set_xlabel(r'$\alpha$ in $rad$')

# create legend
ax_BFE_f[0,0].legend([r'$\mu_\mathrm{f,GP}$', r'$3\sigma$', r'$\mu_\mathrm{f,true}$'])

apply_basic_formatting(fig_BFE_f, width=16, font_size=8)
fig_BFE_f.savefig("VehicleSimulation_APF_muf.pdf", bbox_inches='tight')



plt.show()