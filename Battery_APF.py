import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import pandas as pd

import matplotlib.pyplot as plt

from src.Battery import Battery_APF, basis_fcn, data, forget_factor, offset_C1, offset_R1
from src.Publication_Plotting import plot_BFE_1D, generate_BFE_TimeSlices
from src.Publication_Plotting import apply_basic_formatting, plot_Data



################################################################################
# Model


Sigma_X, Sigma_C1R1, Sigma_Y, weights, Mean_C1R1, Col_Cov_C1R1, Row_Scale_C1R1, df_C1R1 = Battery_APF()



################################################################################
# Plotting

X_pred = np.einsum('...n,...n->...', Sigma_X, weights)
Y_pred = np.einsum('...n,...n->...', Sigma_Y, weights)



# plot data
fig_Y, axes_Y = plot_Data(
    Particles=Sigma_Y[1:],
    weights=weights[1:],
    Reference=data["Voltage"].iloc[1:],
    time=data.index[1:]
)
axes_Y[0].set_ylabel(r"Voltage in $V$")
axes_Y[0].set_xlabel("Time")
apply_basic_formatting(fig_Y, width=8, font_size=8)
fig_Y.savefig("Battery_APF_Fsd.pdf", bbox_inches='tight')



# plot learned function
V = jnp.linspace(0, 2.2, 500)

Mean, Std, X_stats, X_weights, Time = generate_BFE_TimeSlices(
    N_slices=4, 
    X_in=V[...,None], 
    Sigma_X=Sigma_X, 
    Sigma_weights=weights,
    Mean=Mean_C1R1, 
    Col_Scale=Col_Cov_C1R1, 
    Row_Scale=Row_Scale_C1R1, 
    DF=df_C1R1, 
    basis_fcn=basis_fcn, 
    forget_factor=forget_factor
    )


fig_BFE, ax_BFE = plot_BFE_1D(
    V,
    np.array([offset_C1, offset_R1])[None,:,None] + Mean,
    Std,
    Time,
    X_stats, 
    X_weights
)
ax_BFE[0,0].set_ylabel(r"Capacity $C_1$ in $F \times 10^{2}$")
ax_BFE[0,1].set_ylabel(r"Resistance $R_1$ in $\Omega \times 10^{3}$")
for ax in ax_BFE[1]:
    ax.set_xlabel(r"Voltage in $V$")
apply_basic_formatting(fig_BFE, width=16, font_size=8)
fig_BFE.savefig("Battery_APF_Fsd_fcn.pdf", bbox_inches='tight')



plt.show()