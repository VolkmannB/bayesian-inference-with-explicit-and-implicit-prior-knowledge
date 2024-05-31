import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools



from src.Vehicle import features_MTF_front, features_MTF_rear
from src.Vehicle import vehicle_RBF_ip, default_para, f_x_sim, f_y
from src.Vehicle import fx_filter, fy_filter, N_ip
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_updateStatistics, prior_mniw_CondPredictive
from src.Filtering import squared_error, systematic_SISR
from src.VehiclePlotting import generate_Vehicle_Animation



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 200
t_end = 100.0
time = np.arange(0.0, t_end, default_para['dt'])
steps = len(time)


# model prior front tire
GP_prior_f = list(prior_mniw_2naturalPara(
    np.zeros((1, N_ip)),
    np.eye(N_ip),
    np.eye(1)*1e-1,
    0
))

# model prior rear tire
GP_prior_r = list(prior_mniw_2naturalPara(
    np.zeros((1, N_ip)),
    np.eye(N_ip),
    np.eye(1)*1e-1,
    0
))

# parameters for the sufficient statistics
GP_stats_f = [
    np.zeros((N, N_ip, 1)),
    np.zeros((N, N_ip, N_ip)),
    np.zeros((N, 1, 1)),
    np.zeros((N,))
]
GP_stats_r = [
    np.zeros((N, N_ip, 1)),
    np.zeros((N, N_ip, N_ip)),
    np.zeros((N, 1, 1)),
    np.zeros((N,))
]



# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
# np.random.seed(573573)

# noise
R = np.diag([0.01/180*np.pi, 1e-1, 1e-3])
Q = np.diag([5e-4, 5e-4])
R_y = 1e1
w = lambda n=1: np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((R.shape[0],)), R, n)



################################################################################
# Simulation

# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,3))

Sigma_X = np.zeros((steps,N,2))
Sigma_Y = np.zeros((steps,N,2))
Sigma_mu_f = np.zeros((steps,N))
Sigma_mu_r = np.zeros((steps,N))

A_f_Pred = np.zeros((steps, vehicle_RBF_ip.shape[0]))
A_r_Pred = np.zeros((steps, vehicle_RBF_ip.shape[0]))

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
    X[i] = f_x_sim(X[i-1], ctrl_input[i-1], **default_para)
    Y[i] = f_y(X[i], ctrl_input[i], **default_para) + e()
    
    
    
    ####### Filtering
    
    
    ### Step 1: Propagate GP parameters in time
    
    # apply forgetting operator to statistics for t-1 -> t
    GP_stats_f[0] *= 0.999
    GP_stats_f[1] *= 0.999
    GP_stats_f[2] *= 0.999
    GP_stats_f[3] *= 0.999
    GP_stats_r[0] *= 0.999
    GP_stats_r[1] *= 0.999
    GP_stats_r[2] *= 0.999
    GP_stats_r[3] *= 0.999
        
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
        x=Sigma_X[i-1,...],
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
    
    # copy statistics
    GP_stats_f[0] = GP_stats_f[0][idx,...]
    GP_stats_f[1] = GP_stats_f[1][idx,...]
    GP_stats_f[2] = GP_stats_f[2][idx,...]
    GP_stats_f[3] = GP_stats_f[3][idx,...]
    GP_para_f[0] = GP_para_f[0][idx,...]
    GP_para_f[1] = GP_para_f[1][idx,...]
    GP_para_f[2] = GP_para_f[2][idx,...]
    GP_para_f[3] = GP_para_f[3][idx,...]
    
    GP_stats_r[0] = GP_stats_r[0][idx,...]
    GP_stats_r[1] = GP_stats_r[1][idx,...]
    GP_stats_r[2] = GP_stats_r[2][idx,...]
    GP_stats_r[3] = GP_stats_r[3][idx,...]
    GP_para_r[0] = GP_para_r[0][idx,...]
    GP_para_r[1] = GP_para_r[1][idx,...]
    GP_para_r[2] = GP_para_r[2][idx,...]
    GP_para_r[3] = GP_para_r[3][idx,...]
    
    Sigma_X[i-1] = Sigma_X[i-1,idx,...]
    Sigma_mu_f[i-1] = Sigma_mu_f[i-1,idx]
    Sigma_mu_r[i-1] = Sigma_mu_r[i-1,idx]
    
    
    
    ### Step 3: Make a proposal by generating samples from the hirachical 
    # model
    
    # sample from proposal for x at time t
    w_x = w((N,))
    Sigma_X[i] = jax.vmap(
        functools.partial(fx_filter, u=ctrl_input[i-1], **default_para)
        )(
            x=Sigma_X[i-1],
            mu_yf=Sigma_mu_f[i-1],
            mu_yr=Sigma_mu_r[i-1]
            ) + w_x
    
    ## sample proposal for mu front at time t
    # evaluate basis functions
    phi_f0 = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i-1], **default_para)
        )(Sigma_X[i-1])
    phi_f1 = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i], **default_para)
        )(Sigma_X[i])
    
    # calculate conditional predictive distribution
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
        )(
            mean=GP_para_f[0],
            col_cov=GP_para_f[1],
            row_scale=GP_para_f[2],
            df=GP_para_f[3],
            y1=Sigma_mu_f[i-1],
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
        )(Sigma_X[i-1])
    phi_r1 = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[i], **default_para)
        )(Sigma_X[i])
    
    # calculate conditional 
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
        )(
            mean=GP_para_r[0],
            col_cov=GP_para_r[1],
            row_scale=GP_para_r[2],
            df=GP_para_r[3],
            y1=Sigma_mu_r[i-1],
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
        *GP_stats_f,
        Sigma_mu_f[i],
        phi_f1
    ))
    GP_stats_r = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_stats_r,
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
    A_f_Pred[i,...] = np.sum(weights[i,:,None,None] * GP_para_f[0], axis=0).flatten()
    A_r_Pred[i,...] = np.sum(weights[i,:,None,None] * GP_para_r[0], axis=0).flatten()
    
    #abort
    if np.any(np.isnan(weights[i])):
        print("Particle degeneration at new weights")
        break
    
    

fig = generate_Vehicle_Animation(X, Y, ctrl_input, weights, Sigma_X, Sigma_mu_f, Sigma_mu_r, Sigma_Y, A_f_Pred, A_r_Pred, time, default_para, 200., 30., 30)
fig.show()