import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.SingleMassOscillator import F_spring, F_damper, f_x, N_ip, ip, H, m
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics, prior_mniw_sampleLikelihood, gaussian_RBF
from src.Filtering import systematic_SISR, squared_error
from src.SingleMassOscillatorPlotting import generate_Animation



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 200
t_end = 100.0
dt = 0.01
time = np.arange(0.0,t_end,dt)
steps = len(time)

# model of the spring damper system
# parameters of the prior
phi = jax.vmap(H)(ip)
GP_model_prior_eta = list(prior_mniw_2naturalPara(
    np.zeros((1, N_ip)),
    phi@phi.T*40,
    np.eye(1),
    0
))

# parameters for the sufficient statistics
GP_model_stats = [
    np.zeros((N, N_ip, 1)),
    np.zeros((N, N_ip, N_ip)),
    np.zeros((N, 1, 1)),
    np.zeros((N,))
]

# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
seed = np.random.randint(100, 1000000)
print(f"Seed is: {seed}")
np.random.seed(seed)
key = jax.random.key(np.random.randint(100, 1000))
print(f"Initial jax-key is: {key}")

# noise
R = np.array([[1e-3]])
Q = np.diag([5e-6, 5e-7])
w = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), R, n)



################################################################################
# Simulation


# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,))
F_sd = np.zeros((steps,))

Sigma_X = np.zeros((steps,N,2)) # filter
Sigma_F = np.zeros((steps,N))

W = np.zeros((steps, N_ip)) # GP
CW = np.zeros((steps, N_ip, N_ip))

# input
# F = np.ones((steps,)) * -9.81*m + np.sin(2*np.pi*np.arange(0,t_end,dt)/10) * 9.81
F = np.zeros((steps,)) 
F[int(t_end/(5*dt)):] = -9.81*m
F[int(2*t_end/(5*dt)):] = -9.81*m*2
F[int(3*t_end/(5*dt)):] = -9.81*m
F[int(4*t_end/(5*dt)):] = 0

# set initial values
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,)) # initial particles
Sigma_F[0,...] = np.random.normal(0, 1e-6, (N,))
phi = jax.vmap(H)(Sigma_X[0,...])
GP_model_stats = list(jax.vmap(prior_mniw_updateStatistics)( # initial value for GP
    *GP_model_stats,
    Sigma_F[0,...],
    phi
))
X[0,...] = x0
weights = np.ones((steps,N))/N



# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    
    # update system state
    F_sd[i-1] = F_spring(X[i-1,0]) + F_damper(X[i-1,1])
    X[i] = f_x(X[i-1], F[i-1], F_sd[i-1], dt=dt)
    
    # generate measurment
    Y[i] = X[i,0] + e()[0,0]
    
    
    ####### Filtering
    
    ### Step 1: Propagate GP parameters in time
    
    # apply forgetting operator to statistics for t-1 -> t
    GP_model_stats[0] *= 0.999
    GP_model_stats[1] *= 0.999
    GP_model_stats[2] *= 0.999
    GP_model_stats[3] *= 0.999
    
    # calculate parameters of GP from prior and sufficient statistics
    GP_para = list(jax.vmap(prior_mniw_2naturalPara_inv)(
        (GP_model_prior_eta[0] + GP_model_stats[0]),
        (GP_model_prior_eta[1] + GP_model_stats[1]),
        GP_model_prior_eta[2] + GP_model_stats[2],
        GP_model_prior_eta[3] + GP_model_stats[3]
    ))
        
        
        
    ### Step 2: According to the algorithm of the auxiliary PF, resample 
    # particles according to the first stage weights
    
    # create auxiliary variable
    x_aux = jax.vmap(
        functools.partial(f_x, F=F[i-1], dt=dt)
        )(
            x=Sigma_X[i-1,...], 
            F_sd=Sigma_F[i-1,...]
            )
    
    # calculate first stage weights
    l = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(x_aux[:,0,None])
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
    GP_model_stats[0] = GP_model_stats[0][idx,...]
    GP_model_stats[1] = GP_model_stats[1][idx,...]
    GP_model_stats[2] = GP_model_stats[2][idx,...]
    GP_model_stats[3] = GP_model_stats[3][idx,...]
    GP_para[0] = GP_para[0][idx,...]
    GP_para[1] = GP_para[1][idx,...]
    GP_para[2] = GP_para[2][idx,...]
    GP_para[3] = GP_para[3][idx,...]
    
    
    
    ### Step 3: Make a proposal by generating samples from the hirachical 
    # model
    
    # sample from proposal for x at time t
    w_x = w((N,))
    Sigma_X[i] = jax.vmap(
        functools.partial(f_x, F=F[i-1], dt=dt)
        )(
            x=Sigma_X[i-1,idx,:], 
            F_sd=Sigma_F[i-1,idx]
            ) + w_x
    
    # sample from proposal for F at time t
    phi_x0 = jax.vmap(H)(Sigma_X[i-1,idx,:])
    phi_x1 = jax.vmap(H)(Sigma_X[i])
    key, *keys = jax.random.split(key, N+1)
    dphi = phi_x1 - phi_x0
    dxi = jax.vmap(prior_mniw_sampleLikelihood)(
        key=jnp.asarray(keys),
        M=GP_para[0],
        V=GP_para[1],
        Psi=GP_para[2],
        nu=GP_para[3],
        phi=dphi
    ).flatten()
    Sigma_F[i] = Sigma_F[i-1,idx] + dxi
    
    # Update the sufficient statistics of GP with new proposal
    GP_model_stats = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_model_stats,
        Sigma_F[i],
        phi_x1
    ))
    
    # calculate new weights (measurment update)
    sigma_y = Sigma_X[i,:,0,None]
    q = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(sigma_y)
    weights[i] = q / p[idx]
    weights[i] = weights[i]/np.sum(weights[i])
    
    
    
    # logging
    GP_para_logging = prior_mniw_2naturalPara_inv(
        GP_model_prior_eta[0] + np.einsum('n...,n->...', GP_model_stats[0], weights[i]),
        GP_model_prior_eta[1] + np.einsum('n...,n->...', GP_model_stats[1], weights[i]),
        GP_model_prior_eta[2] + np.einsum('n...,n->...', GP_model_stats[2], weights[i]),
        GP_model_prior_eta[3] + np.einsum('n...,n->...', GP_model_stats[3], weights[i])
    )
    W[i,...] = GP_para_logging[0]
    
    #abort
    if np.any(np.isnan(weights[i])):
        print("Particle degeneration at new weights")
        break
    
    

################################################################################
# Plots

fig = generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, H, W, CW, time, 200., 30., 30)

# print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-Sigma_F)))

fig.show()