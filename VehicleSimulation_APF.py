import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools



import matplotlib.pyplot as plt



from src.Vehicle import features_MTF_front, features_MTF_rear, f_alpha
from src.Vehicle import default_para, f_x_sim, f_y, H_vehicle
from src.Vehicle import fx_filter, fy_filter, N_basis_fcn, spectral_density, mu_y
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_updateStatistics, prior_mniw_CondPredictive
from src.Filtering import squared_error, systematic_SISR
from src.Publication_Plotting import plot_BFE_1D, generate_BFE_TimeSlices, plot_Data, apply_basic_formatting



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 200
forget_factor = 0.999
t_end = 100.0
time = np.arange(0.0, t_end, default_para['dt'])
steps = len(time)


# model prior front tire
GP_prior_f = list(prior_mniw_2naturalPara(
    np.zeros((1, N_basis_fcn)),
    np.diag(spectral_density),
    np.eye(1)*1e-1,
    0
))

# model prior rear tire
GP_prior_r = list(prior_mniw_2naturalPara(
    np.zeros((1, N_basis_fcn)),
    np.eye(N_basis_fcn),
    np.eye(1)*1e-1,
    0
))

# parameters for the sufficient statistics
GP_stats_f = [
    np.zeros((N, N_basis_fcn, 1)),
    np.zeros((N, N_basis_fcn, N_basis_fcn)),
    np.zeros((N, 1, 1)),
    np.zeros((N,))
]
GP_stats_r = [
    np.zeros((N, N_basis_fcn, 1)),
    np.zeros((N, N_basis_fcn, N_basis_fcn)),
    np.zeros((N, 1, 1)),
    np.zeros((N,))
]



# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
# np.random.seed(573573)

# noise
R = np.diag([0.01/180*np.pi, 1e-1, 1e-3])
Q = np.diag([1e-9, 1e-9])
R_y = 1e1
w = lambda n=1: np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((R.shape[0],)), R, n)



################################################################################
# Simulation

# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,3))

#variables for logging
Sigma_X = np.zeros((steps,N,2))
Sigma_Y = np.zeros((steps,N,2))
Sigma_mu_f = np.zeros((steps,N))
Sigma_alpha_f = np.zeros((steps,N))
Sigma_mu_r = np.zeros((steps,N))
Sigma_alpha_r = np.zeros((steps,N))

Mean_f = np.zeros((steps, 1, N_basis_fcn))
Col_Cov_f = np.zeros((steps, N_basis_fcn, N_basis_fcn))
Row_Scale_f = np.zeros((steps, 1, 1))
df_f = np.zeros((steps,))

Mean_r = np.zeros((steps, 1, N_basis_fcn))
Col_Cov_r = np.zeros((steps, N_basis_fcn, N_basis_fcn))
Row_Scale_r = np.zeros((steps, 1, 1))
df_r = np.zeros((steps,))

# input
ctrl_input = np.zeros((steps,2))
ctrl_input[:,0] = 10/180*np.pi * np.sin(2*np.pi*time/5) * 0.5*(np.tanh(0.2*(time-15))-np.tanh(0.2*(time-75)))
ctrl_input[:,1] = 11.0


### Set all initial values

# initial values for states
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
Sigma_mu_f[0,...] = np.random.normal(0, 1e-2, (N,))
Sigma_mu_r[0,...] = np.random.normal(0, 1e-2, (N,))
X[0,...] = x0
weights = np.ones((steps,N))/N

# update GP
phi_f = jax.vmap(
    functools.partial(features_MTF_front, u=ctrl_input[0], **default_para)
    )(Sigma_X[0])
phi_r = jax.vmap(
    functools.partial(features_MTF_rear, u=ctrl_input[0], **default_para)
    )(Sigma_X[0])
GP_stats_f = list(jax.vmap(prior_mniw_updateStatistics)(
    *GP_stats_f,
    Sigma_mu_f[0,...],
    phi_f
))
GP_stats_r = list(jax.vmap(prior_mniw_updateStatistics)(
    *GP_stats_r,
    Sigma_mu_r[0,...],
    phi_r
))


# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    X[i] = f_x_sim(X[i-1], ctrl_input[i-1], **default_para) + w()
    Y[i] = f_y(X[i], ctrl_input[i], **default_para) + e()
    
    
    
    ####### Filtering
    
    
    ### Step 1: Propagate GP parameters in time
    
    # apply forgetting operator to statistics for t-1 -> t
    GP_stats_f[0] *= forget_factor
    GP_stats_f[1] *= forget_factor
    GP_stats_f[2] *= forget_factor
    GP_stats_f[3] *= forget_factor
    GP_stats_r[0] *= forget_factor
    GP_stats_r[1] *= forget_factor
    GP_stats_r[2] *= forget_factor
    GP_stats_r[3] *= forget_factor
        
    # calculate parameters of GP from prior and sufficient statistics
    GP_para_f = list(jax.vmap(prior_mniw_2naturalPara_inv)(
        GP_prior_f[0] + GP_stats_f[0],
        GP_prior_f[1] + GP_stats_f[1],
        GP_prior_f[2] + GP_stats_f[2],
        GP_prior_f[3] + GP_stats_f[3]
    ))
    GP_para_r = list(jax.vmap(prior_mniw_2naturalPara_inv)(
        GP_prior_r[0] + GP_stats_r[0],
        GP_prior_r[1] + GP_stats_r[1],
        GP_prior_r[2] + GP_stats_r[2],
        GP_prior_r[3] + GP_stats_r[3]
    ))
        
        
        
    ### Step 2: According to the algorithm of the auxiliary PF, resample 
    # particles according to the first stage weights
    
    # create auxiliary variable for state x
    x_aux = jax.vmap(functools.partial(fx_filter, u=ctrl_input[i-1], **default_para))(
        x=Sigma_X[i-1],
        mu_yf=Sigma_mu_f[i-1], 
        mu_yr=Sigma_mu_r[i-1]
    )
    
    # create auxiliary variable for mu front
    phi_f0 = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i-1], **default_para)
        )(Sigma_X[i-1])
    phi_f1 = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i], **default_para)
        )(x_aux)
    mu_f_aux = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
        )(
            mean=GP_para_f[0],
            col_cov=GP_para_f[1],
            row_scale=GP_para_f[2],
            df=GP_para_f[3],
            y1=Sigma_mu_f[i-1],
            basis1=phi_f0,
            basis2=phi_f1
    )[0]
    
    # create auxiliary variable for mu rear
    phi_r0 = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[i-1], **default_para)
        )(Sigma_X[i-1])
    phi_r1 = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[i], **default_para)
        )(x_aux)
    mu_r_aux = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
        )(
            mean=GP_para_r[0],
            col_cov=GP_para_r[1],
            row_scale=GP_para_r[2],
            df=GP_para_r[3],
            y1=Sigma_mu_r[i-1],
            basis1=phi_r0,
            basis2=phi_r1
    )[0]
    
    # calculate first stage weights
    y_aux = jax.vmap(
        functools.partial(fy_filter, u=ctrl_input[i], **default_para)
        )(
            x=x_aux,
            mu_yf=mu_f_aux, 
            mu_yr=mu_r_aux)
    l = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(x=y_aux)
    p = weights[i-1] * l
    p = p/np.sum(p)
    
    #abort
    if np.any(np.isnan(p)):
        print("Particle degeneration at auxiliary weights")
        break
    
    # draw new indices
    u = np.random.rand()
    idx = np.array(systematic_SISR(u, p))
    idx[idx >= N] = N - 1 # correct out of bounds indices from numerical errors
    
    
    
    ### Step 3: Make a proposal by generating samples from the hirachical 
    # model
    
    # sample from proposal for x at time t
    w_x = w((N,))
    Sigma_X[i] = jax.vmap(
        functools.partial(fx_filter, u=ctrl_input[i-1], **default_para)
        )(
            x=Sigma_X[i-1,idx],
            mu_yf=Sigma_mu_f[i-1,idx],
            mu_yr=Sigma_mu_r[i-1,idx]
            ) + w_x
    
    ## sample proposal for mu front at time t
    # evaluate basis functions
    phi_f0 = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i-1], **default_para)
        )(Sigma_X[i-1,idx])
    phi_f1 = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i], **default_para)
        )(Sigma_X[i])
    
    # calculate conditional predictive distribution
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
        )(
            mean=GP_para_f[0][idx],
            col_cov=GP_para_f[1][idx],
            row_scale=GP_para_f[2][idx],
            df=GP_para_f[3][idx],
            y1=Sigma_mu_f[i-1,idx],
            basis1=phi_f0,
            basis2=phi_f1
    )
        
    # generate samples
    c_col_scale_chol = np.sqrt(np.squeeze(c_col_scale))
    c_row_scale_chol = np.sqrt(np.squeeze(c_row_scale))
    t_samples = np.random.standard_t(df=df)
    Sigma_mu_f[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
    
    ## sample proposal for mu rear at time t
    # evaluate basis fucntions
    phi_r0 = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[i-1], **default_para)
        )(Sigma_X[i-1,idx])
    phi_r1 = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[i], **default_para)
        )(Sigma_X[i])
    
    # calculate conditional 
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
        )(
            mean=GP_para_r[0][idx],
            col_cov=GP_para_r[1][idx],
            row_scale=GP_para_r[2][idx],
            df=GP_para_r[3][idx],
            y1=Sigma_mu_r[i-1,idx],
            basis1=phi_r0,
            basis2=phi_r1
    )
        
    # generate samples
    c_col_scale_chol = np.sqrt(np.squeeze(c_col_scale))
    c_row_scale_chol = np.sqrt(np.squeeze(c_row_scale))
    t_samples = np.random.standard_t(df=df)
    Sigma_mu_r[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
        
        
    
    # Update the sufficient statistics of GP with new proposal
    GP_stats_f = list(jax.vmap(prior_mniw_updateStatistics)(
        GP_stats_f[0][idx],
        GP_stats_f[1][idx],
        GP_stats_f[2][idx],
        GP_stats_f[3][idx],
        Sigma_mu_f[i],
        phi_f1
    ))
    GP_stats_r = list(jax.vmap(prior_mniw_updateStatistics)(
        GP_stats_r[0][idx],
        GP_stats_r[1][idx],
        GP_stats_r[2][idx],
        GP_stats_r[3][idx],
        Sigma_mu_r[i],
        phi_r1
    ))
    
    # calculate new weights
    sigma_y = jax.vmap(
        functools.partial(fy_filter, u=ctrl_input[i], **default_para)
        )(
            x=Sigma_X[i],
            mu_yf=Sigma_mu_f[i], 
            mu_yr=Sigma_mu_r[i])
    Sigma_Y[i] = sigma_y[:,:2]
    q = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(sigma_y)
    weights[i] = q / p[idx]
    weights[i] = weights[i]/np.sum(weights[i])
    
    
    
    # logging
    GP_para_logging = prior_mniw_2naturalPara_inv(
        GP_prior_f[0] + np.einsum('n...,n->...', GP_stats_f[0], weights[i]),
        GP_prior_f[1] + np.einsum('n...,n->...', GP_stats_f[1], weights[i]),
        GP_prior_f[2] + np.einsum('n...,n->...', GP_stats_f[2], weights[i]),
        GP_prior_f[3] + np.einsum('n...,n->...', GP_stats_f[3], weights[i])
    )
    Mean_f[i] = GP_para_logging[0]
    Col_Cov_f[i] = GP_para_logging[1]
    Row_Scale_f[i] = GP_para_logging[2]
    df_f[i] = GP_para_logging[3]
    
    GP_para_logging = prior_mniw_2naturalPara_inv(
        GP_prior_r[0] + np.einsum('n...,n->...', GP_stats_r[0], weights[i]),
        GP_prior_r[1] + np.einsum('n...,n->...', GP_stats_r[1], weights[i]),
        GP_prior_r[2] + np.einsum('n...,n->...', GP_stats_r[2], weights[i]),
        GP_prior_r[3] + np.einsum('n...,n->...', GP_stats_r[3], weights[i])
    )
    Mean_r[i] = GP_para_logging[0]
    Col_Cov_r[i] = GP_para_logging[1]
    Row_Scale_r[i] = GP_para_logging[2]
    df_r[i] = GP_para_logging[3]
    
    Sigma_alpha_f[i], Sigma_alpha_r[i] = jax.vmap(
        functools.partial(f_alpha, **default_para, u=ctrl_input[i])
        )(
        x=Sigma_X[i]
    )
    
    #abort
    if np.any(np.isnan(weights[i])):
        print("Particle degeneration at new weights")
        break



# plot the state estimations
fig_X, axes_X = plot_Data(
    Particles=Sigma_X,
    weights=weights,
    Reference=X,
    time=time
)
apply_basic_formatting(fig_X)

    
    
# plot time slices of the learned MTF for the front tire
alpha = jnp.linspace(-20/180*jnp.pi, 20/180*jnp.pi, 500)
mu_f_true = jax.vmap(
    functools.partial(
        mu_y, 
        mu=default_para['mu'], 
        B=default_para['B_f'], 
        C=default_para['C_f'], 
        E=default_para['E_f']
        )
    )(alpha=alpha)

Mean, Std, X_stats, X_weights, Time = generate_BFE_TimeSlices(
    N_slices=2, 
    X_in=alpha[...,None], 
    Sigma_X=Sigma_alpha_f, 
    Sigma_weights=weights,
    Mean=Mean_f, 
    Col_Scale=Col_Cov_f, 
    Row_Scale=Row_Scale_f, 
    DF=df_f, 
    basis_fcn=H_vehicle, 
    forget_factor=forget_factor
    )

fig_BFE_f, ax_BFE_f = plot_BFE_1D(
    alpha,
    Mean,
    Std,
    Time*default_para['dt'],
    X_stats, 
    X_weights
)

# plot true function
for ax in ax_BFE_f[0]:
    ax.plot(alpha, mu_f_true, color='red', linestyle='--')

# set x label
for ax in ax_BFE_f[1]:
    ax.set_xlabel(r'$\alpha$ in [rad]')

# create legend
ax_BFE_f[0,0].legend([r'$\mu_\mathrm{f,GP}$', r'$3\sigma$', r'$\mu_\mathrm{f,true}$'])

apply_basic_formatting(fig_BFE_f, width=16)



plt.show()