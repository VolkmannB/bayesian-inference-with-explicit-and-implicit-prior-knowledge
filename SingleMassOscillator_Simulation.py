import numpy as np
import jax
import scipy.io



import matplotlib.pyplot as plt



from src.SingleMassOscillator import F_spring, F_damper, basis_fcn, time
from src.SingleMassOscillator import SingleMassOscillator_simulation
from src.SingleMassOscillator import GP_prior, SingleMassOscillator_APF
from src.SingleMassOscillator import SingleMassOscillator_PGAS
from src.BayesianInferrence import prior_mniw_2naturalPara_inv



################################################################################
# Simulation

X, Y, F_sd = SingleMassOscillator_simulation()


################################################################################
### Offline Algorithm

# Run PGAS
(offline_Sigma_X, offline_Sigma_F, 
 offline_weights, offline_GP_stats) = SingleMassOscillator_PGAS(Y)
offline_T0, offline_T1, offline_T2, offline_T3 = offline_GP_stats
del offline_GP_stats



################################################################################
### Online Algorithm

# Run auxiliary Particle filter
(online_Sigma_X, online_Sigma_F, 
 online_weights, online_GP_stats) = SingleMassOscillator_APF(Y)
online_T0, online_T1, online_T2, online_T3 = online_GP_stats
del online_GP_stats
    
    

################################################################################
# Save Results

# precompute input space to function for later plotting
x_plt = np.linspace(-5., 5., 50)
dx_plt = np.linspace(-5., 5., 50)
grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')

X_plot = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
basis_plot = jax.vmap(basis_fcn)(X_plot)

# true spring damper force
F_sd_true_plot = jax.vmap(F_spring)(X_plot[:,0]) + jax.vmap(F_damper)(X_plot[:,1])

# Create save file
mdict = {
    'offline_Sigma_X': offline_Sigma_X,
    'offline_Sigma_F': offline_Sigma_F,
    'offline_weights': offline_weights,
    'offline_T0': offline_T0,
    'offline_T1': offline_T1,
    'offline_T2': offline_T2,
    'offline_T3': offline_T3,
    'online_Sigma_X': online_Sigma_X,
    'online_Sigma_F': online_Sigma_F,
    'online_weights': online_weights,
    'online_T0': online_T0,
    'online_T1': online_T1,
    'online_T2': online_T2,
    'online_T3': online_T3,
    'time': time,
    'X_plot': X_plot,
    'basis_plot': basis_plot,
    'F_sd_true_plot': F_sd_true_plot,
    'prior_T0': GP_prior[0],
    'prior_T1': GP_prior[1],
    'prior_T2': GP_prior[2],
    'prior_T3': GP_prior[3],
    'X': X,
    'F_sd': F_sd
}
scipy.io.savemat('SingleMassOscillator.mat', mdict)
