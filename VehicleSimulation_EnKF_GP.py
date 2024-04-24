import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools



from src.vehicle.Vehicle import features_MTF_front, features_MTF_rear
from src.vehicle.Vehicle import vehicle_RBF_ip, default_para, f_x_sim, f_y
from src.vehicle.Vehicle import fx_filter, fy_filter, f_alpha, mu_y, H_vehicle
from src.RGP import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv
from src.RGP import prior_mniw_updateStatistics, prior_mniw_sampleLikelihood
from src.KalmanFilter import squared_error, systematic_SISR
from src.vehicle.VehiclePlotting import generate_Vehicle_Animation



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 200
N_ip = vehicle_RBF_ip.shape[0]
t_end = 100.0
time = np.arange(0.0, t_end, default_para['dt'])
steps = len(time)


# model prior front tire
phi = jax.vmap(H_vehicle)(vehicle_RBF_ip)
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
np.random.seed(573573)
key = jax.random.key(np.random.randint(100, 1000))
print(f"Initial jax-key is: {key}")

# noise
R = np.diag([0.01/180*np.pi, 1e-1])
Q = np.diag([5e-4, 5e-4])
R_y = 1e-1
w = lambda n=1: np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((R.shape[0],)), R, n)



################################################################################
# Simulation

# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,2))

Sigma_X = np.zeros((steps,N,2))
Sigma_mu_f = np.zeros((steps,N))
Sigma_mu_r = np.zeros((steps,N))

W_f = np.zeros((steps, vehicle_RBF_ip.shape[0]))
CW_f = np.zeros((steps, vehicle_RBF_ip.shape[0], vehicle_RBF_ip.shape[0]))
W_r = np.zeros((steps, vehicle_RBF_ip.shape[0]))
CW_r = np.zeros((steps, vehicle_RBF_ip.shape[0], vehicle_RBF_ip.shape[0]))

# input
ctrl_input = np.zeros((steps,2))
ctrl_input[:,0] = 10/180*np.pi * np.sin(2*np.pi*time/5) * 0.5*(np.tanh(0.2*(time-15))-np.tanh(0.2*(time-75)))
ctrl_input[:,1] = 11.0


### Set all initial values

# initial values for states
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
X[0,...] = x0
weights = np.ones((steps,N))/N

# initial training for GP
phi_f = jax.vmap(
    functools.partial(features_MTF_front, u=ctrl_input[0], **default_para)
    )(Sigma_X[0])
phi_r = jax.vmap(
    functools.partial(features_MTF_rear, u=ctrl_input[0], **default_para)
    )(Sigma_X[0])
GP_stats_f = list(jax.vmap(prior_mniw_updateStatistics)(
    *GP_stats_f,
    np.zeros((N,)),
    phi_f
))
GP_stats_r = list(jax.vmap(prior_mniw_updateStatistics)(
    *GP_stats_r,
    np.zeros((N,)),
    phi_r
))

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

# sample from initial values for mu
key, *keys = jax.random.split(key, N+1)
Sigma_mu_f[0] = jax.vmap(prior_mniw_sampleLikelihood)(
    key=jnp.asarray(keys),
    M=GP_para_f[0],
    V=GP_para_f[1],
    Psi=GP_para_f[2],
    nu=GP_para_f[3],
    phi=phi_f
).flatten()

key, *keys = jax.random.split(key, N+1)
Sigma_mu_r[0] = jax.vmap(prior_mniw_sampleLikelihood)(
    key=jnp.asarray(keys),
    M=GP_para_r[0],
    V=GP_para_r[1],
    Psi=GP_para_r[2],
    nu=GP_para_r[3],
    phi=phi_r
).flatten()


# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    X[i] = f_x_sim(X[i-1], ctrl_input[i-1], **default_para)
    Y[i] = f_y(X[i], ctrl_input[i], **default_para) + e()
    
    
    
    ####### Filtering
    
    # time update
    
    ### Step 1: Evaluate basis functions and calculate standard parameters 
    # of the GP
    
    # evaluate basis functions
    phi_f = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i-1], **default_para)
        )(Sigma_X[i-1])
    phi_r = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[i-1], **default_para)
        )(Sigma_X[i-1])
        
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
    
    # create auxiliary variable
    mu_f_aux = jax.vmap(jnp.matmul)(GP_para_f[0], phi_f).flatten()
    mu_r_aux = jax.vmap(jnp.matmul)(GP_para_r[0], phi_r).flatten()
    x_aux = jax.vmap(functools.partial(fx_filter, u=ctrl_input[i-1], **default_para))(
        x=Sigma_X[i-1,...],
        mu_yf=mu_f_aux, 
        mu_yr=mu_r_aux
    )
    
    # calculate first stage weights
    phi_f_aux_y = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i], **default_para)
        )(x_aux)
    phi_r_aux_y = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[i], **default_para)
        )(x_aux)
    mu_f_aux = jax.vmap(jnp.matmul)(GP_para_f[0], phi_f_aux_y).flatten()
    mu_r_aux = jax.vmap(jnp.matmul)(GP_para_r[0], phi_r_aux_y).flatten()
    y_aux = jax.vmap(
        functools.partial(fy_filter, u=ctrl_input[i], **default_para)
        )(
            x=x_aux,
            mu_yf=mu_f_aux, 
            mu_yr=mu_r_aux)
    l = jax.vmap(functools.partial(squared_error, y=np.hstack((Y[i],0)), cov=np.diag(np.hstack((np.diag(R),R_y)))))(x=y_aux)
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
    phi_f = phi_f[idx]
    GP_stats_r[0] = GP_stats_r[0][idx,...]
    GP_stats_r[1] = GP_stats_r[1][idx,...]
    GP_stats_r[2] = GP_stats_r[2][idx,...]
    GP_stats_r[3] = GP_stats_r[3][idx,...]
    GP_para_r[0] = GP_para_r[0][idx,...]
    GP_para_r[1] = GP_para_r[1][idx,...]
    GP_para_r[2] = GP_para_r[2][idx,...]
    GP_para_r[3] = GP_para_r[3][idx,...]
    phi_r = phi_r[idx]
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
            x=Sigma_X[i-1,:,...],
            mu_yf=Sigma_mu_f[i-1],
            mu_yr=Sigma_mu_r[i-1]
            ) + w_x
    
    # sample from proposal for mu at time t
    phi_f = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[i], **default_para)
        )(Sigma_X[i])
    key, *keys = jax.random.split(key, N+1)
    Sigma_mu_f[i] = jax.vmap(prior_mniw_sampleLikelihood)(
        key=jnp.asarray(keys),
        M=GP_para_f[0],
        V=GP_para_f[1],
        Psi=GP_para_f[2],
        nu=GP_para_f[3],
        phi=phi_f
    ).flatten()
    
    phi_r = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[i], **default_para)
        )(Sigma_X[i])
    key, *keys = jax.random.split(key, N+1)
    Sigma_mu_r[i] = jax.vmap(prior_mniw_sampleLikelihood)(
        key=jnp.asarray(keys),
        M=GP_para_r[0],
        V=GP_para_r[1],
        Psi=GP_para_r[2],
        nu=GP_para_r[3],
        phi=phi_r
    ).flatten()
        
        
    
    ### Step 4: Update the GP
    
    # apply forgetting operator to statistics for t+1
    GP_stats_f[0] *= 0.999
    GP_stats_f[1] *= 0.999
    GP_stats_f[2] *= 0.999
    GP_stats_f[3] *= 0.999
    GP_stats_r[0] *= 0.999
    GP_stats_r[1] *= 0.999
    GP_stats_r[2] *= 0.999
    GP_stats_r[3] *= 0.999
    
    # update GP parameters
    GP_stats_f = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_stats_f,
        Sigma_mu_f[i-1],
        phi_f
    ))
    
    GP_stats_r = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_stats_r,
        Sigma_mu_r[i-1],
        phi_r
    ))
    
    
    
    ### Step 5: Calculate new weights
    
    # calculate new weights (measurment update)
    sigma_y = jax.vmap(
        functools.partial(fy_filter, u=ctrl_input[i], **default_para)
        )(
            x=Sigma_X[i],
            mu_yf=Sigma_mu_f[i], 
            mu_yr=Sigma_mu_r[i])
    q = jax.vmap(functools.partial(squared_error, y=np.hstack((Y[i],0)), cov=np.diag(np.hstack((np.diag(R),R_y)))))(sigma_y)
    weights[i] = q / p[idx]
    weights[i] = weights[i]/np.sum(weights[i])
    
    # logging
    W_f[i,...] = np.sum(weights[i,:,None,None] * GP_para_f[0], axis=0).flatten()
    W_r[i,...] = np.sum(weights[i,:,None,None] * GP_para_r[0], axis=0).flatten()
    
    #abort
    if np.any(np.isnan(weights[i])):
        print("Particle degeneration at new weights")
        break
    
    

fig = generate_Vehicle_Animation(X, Y, ctrl_input, weights, Sigma_X, Sigma_mu_f, Sigma_mu_r, W_f, CW_f, W_r, CW_r, time, default_para, 200., 30., 30)
fig.show()