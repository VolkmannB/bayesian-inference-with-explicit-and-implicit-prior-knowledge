import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import subprocess



import matplotlib.pyplot as plt



from src.Vehicle import Vehicle_simulation, mu_y, steps, basis_fcn
from src.Vehicle import features_MTF_front, features_MTF_rear
from src.Vehicle import time, N_basis_fcn, Vehicle_CPFAS_Kernel
from src.Vehicle import GP_prior_f, GP_prior_r, f_alpha, ctrl_input
from src.Publication_Plotting import plot_BFE_1D
from src.Publication_Plotting import plot_Data, apply_basic_formatting
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_calcStatistics



################################################################################

# Simulation
X, Y = Vehicle_simulation()

### Offline Algorithm
N_iterations = 2
Sigma_X = np.zeros((steps,N_iterations,2))
Sigma_mu_f = np.zeros((steps,N_iterations))
Sigma_mu_r = np.zeros((steps,N_iterations))
weights = np.ones((steps,N_iterations))
Sigma_alpha_f = np.zeros((steps,N_iterations))
Sigma_alpha_r = np.zeros((steps,N_iterations))

# logging of model
# front tire
Mean_f = np.zeros((N_iterations, 1, N_basis_fcn))
Col_Cov_f = np.zeros((N_iterations, N_basis_fcn, N_basis_fcn))
Row_Scale_f = np.zeros((N_iterations, 1, 1))
df_f = np.zeros((N_iterations,))

# rear tire
Mean_r = np.zeros((N_iterations, 1, N_basis_fcn))
Col_Cov_r = np.zeros((N_iterations, N_basis_fcn, N_basis_fcn))
Row_Scale_r = np.zeros((N_iterations, 1, 1))
df_r = np.zeros((N_iterations,))


# variable for sufficient statistics
GP_model_stats_f = [
    np.zeros((N_basis_fcn, 1)),
    np.zeros((N_basis_fcn, N_basis_fcn)),
    np.zeros((1, 1)),
    0
]
GP_model_stats_r = [
    np.zeros((N_basis_fcn, 1)),
    np.zeros((N_basis_fcn, N_basis_fcn)),
    np.zeros((1, 1)),
    0
]


# set initial reference using APF
print(f"\nSetting initial reference trajectory")
GP_para_f = prior_mniw_2naturalPara_inv(
    GP_prior_f[0],
    GP_prior_f[1],
    GP_prior_f[2],
    GP_prior_f[3]
)
GP_para_r = prior_mniw_2naturalPara_inv(
    GP_prior_r[0],
    GP_prior_r[1],
    GP_prior_r[2],
    GP_prior_r[3]
)
Sigma_X[:,0], Sigma_mu_f[:,0], Sigma_mu_r[:,0] = Vehicle_CPFAS_Kernel(
        Y=Y, 
        x_ref=None, 
        mu_f_ref=None, 
        mu_r_ref=None, 
        Mean_f=GP_para_f[0], 
        Col_Cov_f=GP_para_f[1], 
        Row_Scale_f=GP_para_f[2], 
        df_f=GP_para_f[3], 
        Mean_r=GP_para_r[0], 
        Col_Cov_r=GP_para_r[1], 
        Row_Scale_r=GP_para_r[2], 
        df_r=GP_para_r[3]
        )
    
    
# make proposal for distribution of mu_f using new proposals of trajectories
phi = jax.vmap(features_MTF_front)(Sigma_X[:,0], ctrl_input)
T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
        Sigma_mu_f[:,0],
        phi
    )
GP_model_stats_f[0] = np.sum(T_0, axis=0)
GP_model_stats_f[1] = np.sum(T_1, axis=0)
GP_model_stats_f[2] = np.sum(T_2, axis=0)
GP_model_stats_f[3] = np.sum(T_3, axis=0)

# make proposal for distribution of mu_r using new proposals of trajectories
phi = jax.vmap(features_MTF_rear)(Sigma_X[:,0], ctrl_input)
T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
        Sigma_mu_r[:,0],
        phi
    )
GP_model_stats_r[0] = np.sum(T_0, axis=0)
GP_model_stats_r[1] = np.sum(T_1, axis=0)
GP_model_stats_r[2] = np.sum(T_2, axis=0)
GP_model_stats_r[3] = np.sum(T_3, axis=0)

# initial model
GP_para_f = prior_mniw_2naturalPara_inv(
    GP_prior_f[0] + GP_model_stats_f[0],
    GP_prior_f[1] + GP_model_stats_f[1],
    GP_prior_f[2] + GP_model_stats_f[2],
    GP_prior_f[3] + GP_model_stats_f[3]
)
Mean_f[0] = GP_para_f[0]
Col_Cov_f[0] = GP_para_f[1]
Row_Scale_f[0] = GP_para_f[2]
df_f[0] = GP_para_f[3]

GP_para_r = prior_mniw_2naturalPara_inv(
    GP_prior_r[0] + GP_model_stats_r[0],
    GP_prior_r[1] + GP_model_stats_r[1],
    GP_prior_r[2] + GP_model_stats_r[2],
    GP_prior_r[3] + GP_model_stats_r[3]
)
Mean_r[0] = GP_para_r[0]
Col_Cov_r[0] = GP_para_r[1]
Row_Scale_r[0] = GP_para_r[2]
df_r[0] = GP_para_r[3]



### Run PGAS
for k in range(1, N_iterations):
    print(f"\nStarting iteration {k}")
        
    # calculate parameters of GP from prior and sufficient statistics
    GP_para_f = list(prior_mniw_2naturalPara_inv(
        GP_prior_f[0] + GP_model_stats_f[0],
        GP_prior_f[1] + GP_model_stats_f[1],
        GP_prior_f[2] + GP_model_stats_f[2],
        GP_prior_f[3] + GP_model_stats_f[3]
    ))
    print(f"    Var for mu_f is {GP_para_f[2]/(GP_para_f[3]+1)}")
    
    GP_para_r = list(prior_mniw_2naturalPara_inv(
        GP_prior_r[0] + GP_model_stats_r[0],
        GP_prior_r[1] + GP_model_stats_r[1],
        GP_prior_r[2] + GP_model_stats_r[2],
        GP_prior_r[3] + GP_model_stats_r[3]
    ))
    print(f"    Var for mu_r is {GP_para_r[2]/(GP_para_r[3]+1)}")
    
    
    
    # sample new proposal for trajectories using CPF with AS
    Sigma_X[:,k], Sigma_mu_f[:,k], Sigma_mu_r[:,k] = Vehicle_CPFAS_Kernel(
        Y=Y, 
        x_ref=Sigma_X[:,k-1], 
        mu_f_ref=Sigma_mu_f[:,k-1], 
        mu_r_ref=Sigma_mu_r[:,k-1], 
        Mean_f=GP_para_f[0], 
        Col_Cov_f=GP_para_f[1], 
        Row_Scale_f=GP_para_f[2], 
        df_f=GP_para_f[3], 
        Mean_r=GP_para_r[0], 
        Col_Cov_r=GP_para_r[1], 
        Row_Scale_r=GP_para_r[2], 
        df_r=GP_para_r[3]
        )
    Sigma_alpha_f[:,k], Sigma_alpha_r[:,k] = jax.vmap(
        functools.partial(f_alpha)
        )(
        x=Sigma_X[:,k],
        u=ctrl_input
    )
    
    
    # make proposal for distribution of mu_f using new proposals of trajectories
    phi = jax.vmap(features_MTF_front)(Sigma_X[:,k], ctrl_input)
    T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_mu_f[:,k],
            phi
        )
    GP_model_stats_f[0] = np.sum(T_0, axis=0)
    GP_model_stats_f[1] = np.sum(T_1, axis=0)
    GP_model_stats_f[2] = np.sum(T_2, axis=0)
    GP_model_stats_f[3] = np.sum(T_3, axis=0)
    
    # make proposal for distribution of mu_r using new proposals of trajectories
    phi = jax.vmap(features_MTF_rear)(Sigma_X[:,k], ctrl_input)
    T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_mu_r[:,k],
            phi
        )
    GP_model_stats_r[0] = np.sum(T_0, axis=0)
    GP_model_stats_r[1] = np.sum(T_1, axis=0)
    GP_model_stats_r[2] = np.sum(T_2, axis=0)
    GP_model_stats_r[3] = np.sum(T_3, axis=0)
    
    
    # logging
    GP_para_f = prior_mniw_2naturalPara_inv(
        GP_prior_f[0] + GP_model_stats_f[0],
        GP_prior_f[1] + GP_model_stats_f[1],
        GP_prior_f[2] + GP_model_stats_f[2],
        GP_prior_f[3] + GP_model_stats_f[3]
    )
    Mean_f[k] = GP_para_f[0]
    Col_Cov_f[k] = GP_para_f[1]
    Row_Scale_f[k] = GP_para_f[2]
    df_f[k] = GP_para_f[3]
    
    GP_para_r = prior_mniw_2naturalPara_inv(
        GP_prior_r[0] + GP_model_stats_r[0],
        GP_prior_r[1] + GP_model_stats_r[1],
        GP_prior_r[2] + GP_model_stats_r[2],
        GP_prior_r[3] + GP_model_stats_r[3]
    )
    Mean_r[k] = GP_para_r[0]
    Col_Cov_r[k] = GP_para_r[1]
    Row_Scale_r[k] = GP_para_r[2]
    df_r[k] = GP_para_r[3]


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
fig_X.savefig("VehicleSimulation_PGAS_X.pdf", bbox_inches='tight')

    
    
# plot time slices of the learned MTF for the front tire
alpha = jnp.linspace(-20/180*jnp.pi, 20/180*jnp.pi, 500)
mu_f_true = jax.vmap(functools.partial(mu_y))(alpha=alpha)

N_slices = 2
Mean_mu_f = np.zeros((N_slices, *alpha.shape))
Std_mu_f = np.zeros((N_slices, *alpha.shape))
phi_in = jax.vmap(basis_fcn)(alpha)
X_stats = []
X_weights = []
index = (np.array(range(N_slices))+1)/N_slices*(N_iterations-1)
for k in range(N_slices):
    Mean_mu_f[k] = Mean_f[int(index[k])] @ phi_in.T
    Std_mu_f[k] = (
        (np.diag(phi_in @ Col_Cov_f[int(index[k])] @ phi_in.T) + 1) * 
        Row_Scale_f[int(index[k]),0,0] / (df_f[int(index[k])]+1)
        )
    X_stats.append(Sigma_alpha_f[:,int(index[k])])
    X_weights.append(weights[:,int(index[k])])

fig_BFE_f, ax_BFE_f = plot_BFE_1D(
    alpha,
    Mean_mu_f[:,None,...],
    Std_mu_f[:,None,...],
    index,
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
fig_BFE_f.savefig("VehicleSimulation_PGAS_muf.pdf", bbox_inches='tight')



plt.show()