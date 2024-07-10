import numpy as np
import jax



import matplotlib.pyplot as plt



from src.SingleMassOscillator import F_spring, F_damper, basis_fcn, time, steps
from src.SingleMassOscillator import forget_factor, dt
from src.SingleMassOscillator import SingleMassOscillator_simulation
from src.SingleMassOscillator import SingleMassOscillator_APF
from src.Publication_Plotting import plot_BFE_2D, generate_BFE_TimeSlices, plot_Data, apply_basic_formatting



################################################################################

# Simulation
X, Y, F_sd = SingleMassOscillator_simulation()

# Online Algorithm    
Sigma_X, Sigma_F, weights, Mean_F, Col_cov_F, Row_scale_F, df_F = SingleMassOscillator_APF(Y)
    

################################################################################
# Plots


# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=Sigma_X,
    weights=weights,
    Reference=X,
    time=time
)
axes_X[0].set_ylabel(r"$s$ in $m$")
axes_X[1].set_ylabel(r"$\dot{s}$ in $m/s$")
axes_X[1].set_xlabel(r"Time in $s$")
apply_basic_formatting(fig_X, width=8, font_size=8)
fig_X.savefig("SingleMassOscillator_APF_X.pdf", bbox_inches='tight')


# plot the force estimations
fig_F, axes_F = plot_Data(
    Particles=Sigma_F,
    weights=weights,
    Reference=F_sd,
    time=time
)
axes_X[0].set_ylabel(r"$F_\mathrm{sd}$ in $N$")
axes_X[0].set_xlabel(r"Time in $s$")
apply_basic_formatting(fig_F, width=8, font_size=8)
fig_F.savefig("SingleMassOscillator_APF_Fsd.pdf", bbox_inches='tight')



# plot time slices of the learned spring-damper function
x_plt = np.linspace(-5., 5., 50)
dx_plt = np.linspace(-5., 5., 50)
grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')
X_in = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

F_sd_true = jax.vmap(F_spring)(X_in[:,0]) + jax.vmap(F_damper)(X_in[:,1])

Mean, Std, X_stats, X_weights, Time = generate_BFE_TimeSlices(
    N_slices=4, 
    X_in=X_in, 
    Sigma_X=Sigma_X[:int(steps*3/5)], 
    Sigma_weights=weights[:int(steps*3/5)],
    Mean=Mean_F[:int(steps*3/5)], 
    Col_Scale=Col_cov_F[:int(steps*3/5)], 
    Row_Scale=Row_scale_F[:int(steps*3/5)], 
    DF=df_F[:int(steps*3/5)], 
    basis_fcn=basis_fcn, 
    forget_factor=forget_factor
    )

mean_err = np.abs(Mean-F_sd_true)
fig_BFE_F, ax_BFE_F = plot_BFE_2D(
    X_in,
    mean_err,
    Time*dt,
    X_stats, 
    X_weights
)


# set x label
for ax in ax_BFE_F[1]:
    ax.set_xlabel(r'$\dot{s}$ in $m/s$')
ax_BFE_F[0,0].set_ylabel(r'$s$ in $m$')
ax_BFE_F[1,0].set_ylabel(r'$s$ in $m$')
    
apply_basic_formatting(fig_BFE_F, width=16, font_size=8)
fig_BFE_F.savefig("SingleMassOscillator_APF_Fsd_fcn.pdf", bbox_inches='tight')



plt.show()