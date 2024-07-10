import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm
import functools

from src.BayesianInferrence import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_updateStatistics
from src.BayesianInferrence import prior_mniw_CondPredictive
from src.Filtering import systematic_SISR, squared_error


#### This section defines the state space model

# parameters
m=2.0
c1=10.0
c2=2.0
d1=0.7
d2=0.4
C = np.array([[1,0]])


def F_spring(x):
    return c1 * x + c2 * x**3



def F_damper(dx):
    return d1*dx * (1/(1+d2*dx*jnp.tanh(dx)))



def dx(x, F, F_sd, p):
    return jnp.array(
        [x[1], -F_sd/p + F/p + 9.81]
    )



@jax.jit
def f_x(x, F, F_sd, dt, p=m):
    
    # Runge-Kutta 4
    k1 = dx(x, F, F_sd, p)
    k2 = dx(x+dt/2.0*k1, F, F_sd, p)
    k3 = dx(x+dt/2.0*k2, F, F_sd, p)
    k4 = dx(x+dt*k3, F, F_sd, p)
    x = x + dt/6.0*(k1+2*k2+2*k3+k4) 
    
    return x
    
@jax.jit
def f_y(x):
    return C @ x



#### This section defines relevant parameters for the simulation

# set a seed for reproducability
np.random.seed(16723573)

# simulation parameters
N_particles = 200
t_end = 100.0
dt = 0.01
forget_factor = 0.999
time = np.arange(0.0,t_end,dt)
steps = len(time)

# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])

# noise
R = np.array([[1e-3]])
Q = np.diag([5e-8, 5e-9])
w = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), R, n)

# external force
F_ext = np.zeros((steps,)) 
F_ext[int(t_end/(5*dt)):] = -9.81*m
F_ext[int(2*t_end/(5*dt)):] = -9.81*m*2
F_ext[int(3*t_end/(5*dt)):] = -9.81*m
F_ext[int(4*t_end/(5*dt)):] = 0
    
    
    
#### This section defines the basis function expansion

# basis functions
N_basis_fcn = 41
basis_fcn, sd = generate_Hilbert_BasisFunction(
    num_fcn=N_basis_fcn, 
    domain_boundary=np.array([[-7.5, 7.5],[-7.5, 7.5]]), 
    lengthscale=7.5*2/N_basis_fcn, 
    scale=60
    )



# parameters of the MNIW prior
GP_model_prior = list(prior_mniw_2naturalPara(
    np.zeros((1, N_basis_fcn)),
    np.diag(sd),
    np.eye(1)*10,
    0
))



#### This section defines a function for the simulation of the system

def SingleMassOscillator_simulation():
        
    # time series for plot
    X = np.zeros((steps,2)) # sim
    Y = np.zeros((steps,))
    F_sd = np.zeros((steps,))
    
    # set initial value
    X[0,...] = x0
    
        
    # simulation loop
    for i in tqdm(range(1,steps), desc="Running System Simulation"):
        
        # update system state
        F_sd[i-1] = F_spring(X[i-1,0]) + F_damper(X[i-1,1])
        X[i] = f_x(X[i-1], F_ext[i-1], F_sd[i-1], dt=dt) + w()
        
        # generate measurment
        Y[i] = X[i,0] + e()[0,0]
    
    return X, Y, F_sd



#### This section defines a function to run the online version of the algorithm

def SingleMassOscillator_APF(Y):
    
    # Particle trajectories
    Sigma_X = np.zeros((steps,N_particles,2))
    Sigma_F = np.zeros((steps,N_particles))
    weights = np.ones((steps,N_particles))/N_particles
    
    # logging of model
    Mean_F = np.zeros((steps, 1, N_basis_fcn)) # GP
    Col_cov_F = np.zeros((steps, N_basis_fcn, N_basis_fcn)) # GP
    Row_scale_F = np.zeros((steps, 1, 1)) # GP
    df_F = np.zeros((steps,)) # GP
    
    # variable for the sufficient statistics
    GP_model_stats = [
        np.zeros((N_particles, N_basis_fcn, 1)),
        np.zeros((N_particles, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_particles, 1, 1)),
        np.zeros((N_particles,))
    ]
    
    ## set initial values
    # initial particles
    Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N_particles,)) 
    Sigma_F[0,...] = np.random.normal(0, 1e-6, (N_particles,))
    phi = jax.vmap(basis_fcn)(Sigma_X[0,...])
    
    # initial value for GP
    GP_model_stats = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_model_stats,
        Sigma_F[0,...],
        phi
    ))
    
    
    
    for i in tqdm(range(1,steps), desc="Running Online Algorithm"):
        
        ### Step 1: Propagate GP parameters in time
        
        # apply forgetting operator to statistics for t-1 -> t
        GP_model_stats[0] *= forget_factor
        GP_model_stats[1] *= forget_factor
        GP_model_stats[2] *= forget_factor
        GP_model_stats[3] *= forget_factor
        
        # calculate parameters of GP from prior and sufficient statistics
        GP_para = list(jax.vmap(prior_mniw_2naturalPara_inv)(
            GP_model_prior[0] + GP_model_stats[0],
            GP_model_prior[1] + GP_model_stats[1],
            GP_model_prior[2] + GP_model_stats[2],
            GP_model_prior[3] + GP_model_stats[3]
        ))
            
            
            
        ### Step 2: According to the algorithm of the auxiliary PF, resample 
        # particles according to the first stage weights
        
        # create auxiliary variable
        x_aux = jax.vmap(
            functools.partial(f_x, F=F_ext[i-1], dt=dt)
            )(
                x=Sigma_X[i-1], 
                F_sd=Sigma_F[i-1]
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
        idx[idx >= N_particles] = N_particles - 1 # correct out of bounds indices from numerical errors
        
        
        
        ### Step 3: Make a proposal by generating samples from the hirachical 
        # model
        
        # sample from proposal for x at time t
        w_x = w((N_particles,))
        Sigma_X[i] = jax.vmap(
            functools.partial(f_x, F=F_ext[i-1], dt=dt)
            )(
                x=Sigma_X[i-1,idx], 
                F_sd=Sigma_F[i-1,idx]
        ) + w_x
        
        ## sample from proposal for F at time t
        # evaluate basis functions for all particles
        phi_x0 = jax.vmap(basis_fcn)(Sigma_X[i-1,idx])
        phi_x1 = jax.vmap(basis_fcn)(Sigma_X[i])
        
        # calculate conditional predictive distribution
        c_mean, c_col_scale, c_row_scale, df = jax.vmap(
            functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
            )(
                mean=GP_para[0][idx],
                col_cov=GP_para[1][idx],
                row_scale=GP_para[2][idx],
                df=GP_para[3][idx],
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
            GP_model_stats[0][idx],
            GP_model_stats[1][idx],
            GP_model_stats[2][idx],
            GP_model_stats[3][idx],
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
            GP_model_prior[0] + np.einsum('n...,n->...', GP_model_stats[0], weights[i]),
            GP_model_prior[1] + np.einsum('n...,n->...', GP_model_stats[1], weights[i]),
            GP_model_prior[2] + np.einsum('n...,n->...', GP_model_stats[2], weights[i]),
            GP_model_prior[3] + np.einsum('n...,n->...', GP_model_stats[3], weights[i])
        )
        Mean_F[i] = GP_para_logging[0]
        Col_cov_F[i] = GP_para_logging[1]
        Row_scale_F[i] = GP_para_logging[2]
        df_F[i] = GP_para_logging[3]
        
        #abort
        if np.any(np.isnan(weights[i])):
            print("Particle degeneration at new weights")
            break
        
    return Sigma_X, Sigma_F, weights, Mean_F, Col_cov_F, Row_scale_F, df_F