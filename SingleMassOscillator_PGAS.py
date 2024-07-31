import numpy as np
import jax
import scipy.stats



import matplotlib.pyplot as plt



from src.SingleMassOscillator import F_spring, F_damper, basis_fcn, steps, time
from src.SingleMassOscillator import SingleMassOscillator_simulation
from src.SingleMassOscillator import SingleMassOscillator_CPFAS_Kernel
from src.SingleMassOscillator import SingleMassOscillator_APF
from src.SingleMassOscillator import GP_model_prior, N_basis_fcn
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_calcStatistics
from src.Publication_Plotting import plot_BFE_2D, apply_basic_formatting, plot_Data



################################################################################

# Simulation
X, Y, F_sd = SingleMassOscillator_simulation()

### Online Algorithm
N_iterations = 10
Sigma_X = np.zeros((steps,N_iterations,2))
Sigma_F = np.zeros((steps,N_iterations))
Likelihood_Y = np.zeros((N_iterations,))
weights = np.ones((steps,N_iterations))/N_iterations * np.flip(np.cumprod(np.ones((N_iterations))*0.5))
weights = weights / np.sum(weights[0])

# logging of model
Mean_F = np.zeros((N_iterations, 1, N_basis_fcn)) # GP
Col_cov_F = np.zeros((N_iterations, N_basis_fcn, N_basis_fcn)) # GP
Row_scale_F = np.zeros((N_iterations, 1, 1)) # GP
df_F = np.zeros((N_iterations,)) # GP

# variable for sufficient statistics
GP_model_stats = [
    np.zeros((N_basis_fcn, 1)),
    np.zeros((N_basis_fcn, N_basis_fcn)),
    np.zeros((1, 1)),
    0
]

# variable for sufficient statistics - logging over iterations
GP_model_stats_logging = [
    np.zeros((N_basis_fcn, 1)),
    np.zeros((N_basis_fcn, N_basis_fcn)),
    np.zeros((1, 1)),
    0
]

# set initial reference using APF
print(f"\nSetting initial reference trajectory")
sample_X, sample_F, sample_w, _, _, _, _ = SingleMassOscillator_APF(Y)
Sigma_X[:,0] = np.einsum('in...,in->i...', sample_X, sample_w)
Sigma_F[:,0] = np.einsum('in...,in->i...', sample_F, sample_w)

# initial model
GP_para_logging = prior_mniw_2naturalPara_inv(
    GP_model_prior[0] + GP_model_stats_logging[0],
    GP_model_prior[1] + GP_model_stats_logging[1],
    GP_model_prior[2] + GP_model_stats_logging[2],
    GP_model_prior[3] + GP_model_stats_logging[3]
)
Mean_F[0] = GP_para_logging[0]
Col_cov_F[0] = GP_para_logging[1]
Row_scale_F[0] = GP_para_logging[2]
df_F[0] = GP_para_logging[3]



### Run PGAS
for k in range(1, N_iterations):
    print(f"\nStarting iteration {k}")
        
    # calculate parameters of GP from prior and sufficient statistics
    GP_para = list(prior_mniw_2naturalPara_inv(
        GP_model_prior[0] + GP_model_stats[0],
        GP_model_prior[1] + GP_model_stats[1],
        GP_model_prior[2] + GP_model_stats[2],
        GP_model_prior[3] + GP_model_stats[3]
    ))
    print(f"    Var is {GP_para[2]/(GP_para[3]+1)}")
    
    
    
    # sample new proposal for trajectories using CPF with AS
    Sigma_X[:,k], Sigma_F[:,k] = SingleMassOscillator_CPFAS_Kernel(
        Y=Y,
        x_ref=Sigma_X[:,k-1],
        F_ref=Sigma_F[:,k-1],
        Mean_F=GP_para[0], 
        Col_cov_F=GP_para[1], 
        Row_scale_F=GP_para[2], 
        df_F=GP_para[3])
    
    
    # make proposal for distribution of F_sd using new proposals of trajectories
    phi = jax.vmap(basis_fcn)(Sigma_X[:,k])
    T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_F[:,k],
            phi
        )
    GP_model_stats[0] = np.sum(T_0, axis=0)
    GP_model_stats[1] = np.sum(T_1, axis=0)
    GP_model_stats[2] = np.sum(T_2, axis=0)
    GP_model_stats[3] = np.sum(T_3, axis=0)
    
    
    # logging
    GP_model_stats_logging[0] = GP_model_stats_logging[0] * 0.5 + np.sum(T_0, axis=0)
    GP_model_stats_logging[1] = GP_model_stats_logging[1] * 0.5 + np.sum(T_1, axis=0)
    GP_model_stats_logging[2] = GP_model_stats_logging[2] * 0.5 + np.sum(T_2, axis=0)
    GP_model_stats_logging[3] = GP_model_stats_logging[3] * 0.5 + np.sum(T_3, axis=0)
    
    GP_para_logging = prior_mniw_2naturalPara_inv(
        GP_model_prior[0] + GP_model_stats_logging[0],
        GP_model_prior[1] + GP_model_stats_logging[1],
        GP_model_prior[2] + GP_model_stats_logging[2],
        GP_model_prior[3] + GP_model_stats_logging[3]
    )
    Mean_F[k] = GP_para_logging[0]
    Col_cov_F[k] = GP_para_logging[1]
    Row_scale_F[k] = GP_para_logging[2]
    df_F[k] = GP_para_logging[3]
    

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

N_slices = 4
Mean = np.zeros((N_slices, *F_sd_true.shape))
phi_in = jax.vmap(basis_fcn)(X_in)
X_stats = []
X_weights = []
for k in range(N_slices):
    index = int((k+1)/N_slices*(N_iterations-1))
    Mean[k] = Mean_F[index] @ phi_in.T
    X_stats.append(Sigma_X[:,:index])
    X_weights.append(weights[:,:index])


mean_err = np.abs(Mean-F_sd_true)[:,None,:]
fig_BFE_F, ax_BFE_F = plot_BFE_2D(
    X_in,
    mean_err,
    np.array(range(N_slices)),
    X_stats, 
    X_weights
)
for k in range(len(ax_BFE_F[0])):
    ax_BFE_F[0,k].set_title(f"Iteration {int((k+1)/N_slices*(N_iterations-1))}")


# set x label
for ax in ax_BFE_F[1]:
    ax.set_xlabel(r'$\dot{s}$ in $m/s$')
ax_BFE_F[0,0].set_ylabel(r'$s$ in $m$')
ax_BFE_F[1,0].set_ylabel(r'$s$ in $m$')
    
apply_basic_formatting(fig_BFE_F, width=16, font_size=8)
fig_BFE_F.savefig("SingleMassOscillator_APF_Fsd_fcn.pdf", bbox_inches='tight')



plt.show()