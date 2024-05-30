import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools

import plotly.graph_objects as go
from plotly.subplots import make_subplots



from src.SingleMassOscillator import F_spring, F_damper, f_x, N_ip, ip, H, m
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_updateStatistics, prior_mniw_CondPredictive
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
    np.eye(N_ip)*40,
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
# np.random.seed(573573)

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

A_Pred = np.zeros((steps, N_ip)) # GP

N_eff = np.zeros((steps,)) # effective sample size

# input
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
        GP_model_prior_eta[0] + GP_model_stats[0],
        GP_model_prior_eta[1] + GP_model_stats[1],
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
    
    ## sample from proposal for F at time t
    # evaluate basis functions for all particles
    phi_x0 = jax.vmap(H)(Sigma_X[i-1,idx,:])
    phi_x1 = jax.vmap(H)(Sigma_X[i])
    
    # calculate conditional predictive distribution
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
        )(
            mean=GP_para[0],
            col_cov=GP_para[1],
            row_scale=GP_para[2],
            df=GP_para[3],
            y1=Sigma_F[i-1,idx],
            basis1=phi_x0,
            basis2=phi_x1
    )
    
    # generate samples
    c_col_scale_chol = np.sqrt(np.squeeze(c_col_scale))
    c_row_scale_chol = np.sqrt(np.squeeze(c_row_scale))
    t_samples = np.random.standard_t(df=df)
    Sigma_F[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
    
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
    N_eff[i] = 1/np.sum(weights[i]**2)/N
    
    
    
    # logging
    GP_para_logging = prior_mniw_2naturalPara_inv(
        GP_model_prior_eta[0] + np.einsum('n...,n->...', GP_model_stats[0], weights[i]),
        GP_model_prior_eta[1] + np.einsum('n...,n->...', GP_model_stats[1], weights[i]),
        GP_model_prior_eta[2] + np.einsum('n...,n->...', GP_model_stats[2], weights[i]),
        GP_model_prior_eta[3] + np.einsum('n...,n->...', GP_model_stats[3], weights[i])
    )
    A_Pred[i,...] = GP_para_logging[0]
    
    #abort
    if np.any(np.isnan(weights[i])):
        print("Particle degeneration at new weights")
        break
    
    

################################################################################
# Plots

# genearte animation
fig = generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, H, A_Pred, time, 200., 30., 30)
fig.show()

# plot distribution of F_sd
fig = make_subplots()
time = time[0:-1:5]
N_eff = N_eff[0:-1:5]
Sigma_F = Sigma_F[0:-1:5,0:-1:5]
weights = weights[0:-1:5,0:-1:5]
weights = weights/np.sum(weights, axis=-1, keepdims=True)
for i in range(weights.shape[-1]):
    fig.add_trace(
        go.Scatter(
            x=time,
            y=Sigma_F[:,i],
            mode='markers',
            marker={
                'color': 'blue', 
                'size':8,
                'opacity': weights[:,i]
            }
        )
    )
fig.update_layout(showlegend=False)
fig.show()

# plot a single sample of F_sd over time
fig = make_subplots()
fig.add_trace(
    go.Scatter(
        x=time,
        y=Sigma_F[:,0],
        mode='lines',
        line={
            'color': 'blue', 
            'width':3
        }
    )
)
fig.update_layout(showlegend=False)
fig.show()

# plot effective sample size over time
fig = make_subplots()
fig.add_trace(
    go.Scatter(
        x=time,
        y=N_eff,
        mode='lines',
        line={
            'color': 'blue', 
            'width':3
        }
    )
)
fig.update_layout(showlegend=False)
fig.show()