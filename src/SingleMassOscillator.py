import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm
import functools
import scipy
import jax.scipy as jsc

from src.BayesianInferrence import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_calcStatistics
from src.BayesianInferrence import prior_mniw_CondPredictive
from src.Filtering import systematic_SISR, log_likelihood_Normal, log_likelihood_Multivariate_t


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



@jax.jit
def log_likelihood(obs_x, obs_F, x_mean, x_1, F_1, Mean_F, Col_cov_F, Row_scale_F, df_F):
    
    # log likelihood of state x
    l_x = log_likelihood_Normal(obs_x, x_mean, Q)
    
    # log likelihood of force F_sd from conditional predictive PDF
    phi_1 = basis_fcn(x_1)
    phi_2 = basis_fcn(obs_x)
    c_mean, c_col_scale, c_row_scale, c_df = prior_mniw_CondPredictive(
        y1_var=y1_var,
        mean=Mean_F,
        col_cov=Col_cov_F,
        row_scale=Row_scale_F,
        df=df_F,
        y1=F_1,
        basis1=phi_1,
        basis2=phi_2
        )
    
    l_F = log_likelihood_Multivariate_t(
        observed=obs_F, 
        mean=c_mean, 
        scale=c_col_scale*c_row_scale, 
        df=c_df
        )
    
    return l_x + l_F



#### This section defines relevant parameters for the simulation

# set a seed for reproducability
rng = np.random.default_rng(16723573)

# simulation parameters
N_particles = 200
N_PGAS_iter = 200
t_end = 100.0
dt = 0.01
forget_factor = 0.999
time = np.arange(0.0,t_end,dt)
steps = len(time)
y1_var=3e-2

# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
P0_F = 1e-6

# noise
R = np.array([[1e-3]])
Q = np.diag([5e-8, 5e-9])
w = lambda n=1: rng.multivariate_normal(np.zeros((2,)), Q, n)
e = lambda n=1: rng.multivariate_normal(np.zeros((1,)), R, n)

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
GP_prior = list(prior_mniw_2naturalPara(
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
    
    print("\n=== Online Algorithm ===")
    
    # Particle trajectories
    Sigma_X = np.zeros((steps, N_particles, 2))
    Sigma_F = np.zeros((steps, N_particles))
    weights = np.ones((steps, N_particles))/N_particles
    
    # variable for the sufficient statistics
    GP_stats = [
        np.zeros((N_particles, N_basis_fcn, 1)),
        np.zeros((N_particles, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_particles, 1, 1)),
        np.zeros((N_particles,))
    ]
    GP_stats_logging = [
        np.zeros((steps, N_basis_fcn, 1)),
        np.zeros((steps, N_basis_fcn, N_basis_fcn)),
        np.zeros((steps, 1, 1)),
        np.zeros((steps,))
    ]
    
    ## set initial values
    # initial particles
    Sigma_X[0,...] = rng.multivariate_normal(x0, P0, (N_particles,)) 
    Sigma_F[0,...] = rng.normal(0, P0_F, (N_particles,))
    phi = jax.vmap(basis_fcn)(Sigma_X[0,...])
    
    # initial value for GP
    T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
        Sigma_F[0,...],
        phi
    )
    GP_stats[0] += T_0
    GP_stats[1] += T_1
    GP_stats[2] += T_2
    GP_stats[3] += T_3
    
    # logging
    GP_stats_logging[0][0] = np.einsum('n...,n->...', T_0, weights[0])
    GP_stats_logging[1][0] = np.einsum('n...,n->...', T_1, weights[0])
    GP_stats_logging[2][0] = np.einsum('n...,n->...', T_2, weights[0])
    GP_stats_logging[3][0] = np.einsum('n...,n->...', T_3, weights[0])
    
    
    
    for i in tqdm(range(1,steps), desc="Running APF Algorithm"):
        
        ### Step 1: Propagate GP parameters in time
        
        # apply forgetting operator to statistics for t-1 -> t
        GP_stats[0] *= forget_factor
        GP_stats[1] *= forget_factor
        GP_stats[2] *= forget_factor
        GP_stats[3] *= forget_factor
        
        # calculate parameters of GP from prior and sufficient statistics
        GP_para = list(jax.vmap(prior_mniw_2naturalPara_inv)(
            GP_prior[0] + GP_stats[0],
            GP_prior[1] + GP_stats[1],
            GP_prior[2] + GP_stats[2],
            GP_prior[3] + GP_stats[3]
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
        l_y_aux = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(x_aux[:,0,None])
        weights_aux = weights[i-1] * np.exp(l_y_aux)
        weights_aux = weights_aux/np.sum(weights_aux)
        
        #abort
        if np.any(np.isnan(weights_aux)):
            print("Particle degeneration at auxiliary weights")
            break
        
        # draw new indices
        u = rng.random()
        idx = np.array(systematic_SISR(u, weights_aux))
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
            functools.partial(prior_mniw_CondPredictive, y1_var=y1_var)
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
        t_samples = rng.standard_t(df=df)
        Sigma_F[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
        
        # Update the sufficient statistics of GP with new proposal
        T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_F[i],
            phi_x1
        )
        GP_stats[0] = GP_stats[0][idx] + T_0
        GP_stats[1] = GP_stats[1][idx] + T_1
        GP_stats[2] = GP_stats[2][idx] + T_2
        GP_stats[3] = GP_stats[3][idx] + T_3
        
        # calculate new weights (measurment update)
        sigma_y = Sigma_X[i,:,0,None]
        l_y = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(sigma_y)
        weights[i] = np.exp(l_y - l_y_aux[idx])
        weights[i] = weights[i]/np.sum(weights[i])
        
        
        
        # logging
        GP_stats_logging[0][i] = np.einsum('n...,n->...', T_0, weights[i])
        GP_stats_logging[1][i] = np.einsum('n...,n->...', T_1, weights[i])
        GP_stats_logging[2][i] = np.einsum('n...,n->...', T_2, weights[i])
        GP_stats_logging[3][i] = np.einsum('n...,n->...', T_3, weights[i])
        
        #abort
        if np.any(np.isnan(weights[i])):
            print("Particle degeneration at new weights")
            break
        
    return Sigma_X, Sigma_F, weights, GP_stats_logging



#### This section defines a function to run the offline version of the algorithm

def SingleMassOscillator_PGAS(Y):
    """
    This function runs the PGAS algorithm.

    Args:
        Y (Array): Measurements
    """
    print("\n=== Offline Algorithm ===")
    
    Sigma_X = np.zeros((steps,N_PGAS_iter,2))
    Sigma_F = np.zeros((steps,N_PGAS_iter))
    weights = np.ones((steps,N_PGAS_iter))/N_PGAS_iter

    # variable for sufficient statistics
    GP_stats = [
        np.zeros((N_PGAS_iter, N_basis_fcn, 1)),
        np.zeros((N_PGAS_iter, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_PGAS_iter, 1, 1)),
        np.zeros((N_PGAS_iter,))
    ]

    # set initial reference using CPFAS Kernel without reference
    print(f"Setting initial reference trajectory")
    GP_para = prior_mniw_2naturalPara_inv(
        GP_prior[0],
        GP_prior[1],
        GP_prior[2],
        GP_prior[3]
    )
    Sigma_X[:,0], Sigma_F[:,0] = SingleMassOscillator_CPFAS_Kernel(
            Y=Y,
            x_ref=None,
            F_ref=None,
            Mean_F=GP_para[0], 
            Col_Cov_F=GP_para[1], 
            Row_Scale_F=GP_para[2], 
            df_F=GP_para[3])
        
        
    # make proposal for distribution of F_sd using new proposals of trajectories
    phi = jax.vmap(basis_fcn)(Sigma_X[:,0])
    T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_F[:,0],
            phi
        )
    GP_stats[0][0] = np.sum(T_0, axis=0)
    GP_stats[1][0] = np.sum(T_1, axis=0)
    GP_stats[2][0] = np.sum(T_2, axis=0)
    GP_stats[3][0] = np.sum(T_3, axis=0)



    ### Run PGAS
    for k in range(1, N_PGAS_iter):
        print(f"Starting iteration {k}")
            
        # calculate parameters of GP from prior and sufficient statistics
        GP_para = list(prior_mniw_2naturalPara_inv(
            GP_prior[0] + GP_stats[0][k-1],
            GP_prior[1] + GP_stats[1][k-1],
            GP_prior[2] + GP_stats[2][k-1],
            GP_prior[3] + GP_stats[3][k-1]
        ))
        
        
        
        # sample new proposal for trajectories using CPF with AS
        Sigma_X[:,k], Sigma_F[:,k] = SingleMassOscillator_CPFAS_Kernel(
            Y=Y,
            x_ref=Sigma_X[:,k-1],
            F_ref=Sigma_F[:,k-1],
            Mean_F=GP_para[0], 
            Col_Cov_F=GP_para[1], 
            Row_Scale_F=GP_para[2], 
            df_F=GP_para[3])
        
        
        # make proposal for distribution of F_sd using new proposals of trajectories
        phi = jax.vmap(basis_fcn)(Sigma_X[:,k])
        T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
                Sigma_F[:,k],
                phi
            )
        GP_stats[0][k] = np.sum(T_0, axis=0)
        GP_stats[1][k] = np.sum(T_1, axis=0)
        GP_stats[2][k] = np.sum(T_2, axis=0)
        GP_stats[3][k] = np.sum(T_3, axis=0)
    
    
    return Sigma_X, Sigma_F, weights, GP_stats



def SingleMassOscillator_CPFAS_Kernel(
    Y, 
    x_ref, 
    F_ref, 
    Mean_F, 
    Col_Cov_F, 
    Row_Scale_F, 
    df_F
    ):
    """
    This function runs the CPFAS kernel for the PGAS algorithm to generate new 
    proposals for the state trajectory. It is a conditional auxiliary particle 
    filter. 
    
    """
    
    # Particle trajectories
    Sigma_X = np.zeros((steps,N_particles,2))
    Sigma_F = np.zeros((steps,N_particles))
    weights = np.ones((steps,N_particles))/N_particles
    ancestor_idx = np.zeros((steps-1,N_particles))
    
    ## set initial values
    Sigma_X[0,...] = rng.multivariate_normal(x0, P0, (N_particles,)) 
    Sigma_F[0,...] = rng.normal(0, P0_F, (N_particles,))
    
    if x_ref is not None:
        Sigma_X[0,-1] = x_ref[0]
        Sigma_F[0,-1] = F_ref[0]
    
    
    
    for i in tqdm(range(1,steps), desc="    Running CPF Kernel"):
        
        ### Step 1: According to the algorithm of the auxiliary PF, draw new 
        # ancestor indices according to the first stage weights
        
        # create auxiliary variable
        x_aux = jax.vmap(
            functools.partial(f_x, F=F_ext[i-1], dt=dt)
            )(
                x=Sigma_X[i-1], 
                F_sd=Sigma_F[i-1]
                )
        
        # calculate first stage weights
        l_y_aux = jax.vmap(
            functools.partial(log_likelihood_Normal, mean=Y[i], cov=R)
            )(x_aux[:,0,None])
        weights_aux = weights[i-1] * np.exp(l_y_aux)
        weights_aux = weights_aux/np.sum(weights_aux)
        
        #abort
        if np.any(np.isnan(weights_aux)):
            print("Particle degeneration at auxiliary weights")
            break
        
        # draw new indices
        u = rng.random()
        idx = np.array(systematic_SISR(u, weights_aux))
        # correct out of bounds indices from numerical errors
        idx[idx >= N_particles] = N_particles - 1 
        
        # let reference trajectory survive
        if x_ref is not None:
            idx[-1] = N_particles - 1
        
        
        
        ### Step 2: Make a proposal by generating samples from the hirachical 
        # model
        
        # sample from proposal for x at time t
        w_x = w((N_particles,))
        Sigma_X_mean = jax.vmap(
            functools.partial(f_x, F=F_ext[i-1], dt=dt)
            )(
                x=Sigma_X[i-1,idx], 
                F_sd=Sigma_F[i-1,idx]
        )
        Sigma_X[i] = Sigma_X_mean + w_x
        
        # set reference trajectory for state x
        if x_ref is not None:
            Sigma_X[i,-1] = x_ref[i]
        
        ## sample from proposal for F at time t
        # evaluate basis functions for all particles
        phi_x0 = jax.vmap(basis_fcn)(Sigma_X[i-1,idx])
        phi_x1 = jax.vmap(basis_fcn)(Sigma_X[i])
        
        # calculate conditional predictive distribution
        c_mean, c_col_scale, c_row_scale, df = jax.vmap(
            functools.partial(
                prior_mniw_CondPredictive, 
                y1_var=y1_var,
                mean=Mean_F,
                col_cov=Col_Cov_F,
                row_scale=Row_Scale_F,
                df=df_F
                )
            )(
            y1=Sigma_F[i-1,idx],
            basis1=phi_x0,
            basis2=phi_x1
        )
        
        # generate samples
        c_col_scale_chol = np.sqrt(np.squeeze(c_col_scale))
        c_row_scale_chol = np.sqrt(np.squeeze(c_row_scale))
        t_samples = rng.standard_t(df=df)
        Sigma_F[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
        
        # set reference trajectory for F_sd
        if x_ref is not None:
            Sigma_F[i,-1] = F_ref[i]
        
        
        
        ### Step 3: Sample a new ancestor for the reference trajectory
        
        if x_ref is not None:
            
            # calculate ancestor weights
            l_x = jax.vmap(
                functools.partial(
                    log_likelihood, 
                    obs_x=Sigma_X[i,-1], 
                    obs_F=Sigma_F[i,-1], 
                    Mean_F=Mean_F, 
                    Col_cov_F=Col_Cov_F, 
                    Row_scale_F=Row_Scale_F, 
                    df_F=df_F)
                )(x_mean=Sigma_X_mean, x_1=Sigma_X[i-1], F_1=Sigma_F[i-1])
            weights_ancestor = weights[i-1] * np.exp(l_x) # un-normalized
            
            # sample an ancestor index for reference trajectory
            if np.isclose(np.sum(weights_ancestor), 0):
                ref_idx = N_particles - 1
            else:
                weights_ancestor /= np.sum(weights_ancestor)
                u = rng.random()
                ref_idx = np.searchsorted(np.cumsum(weights_ancestor), u)
            
            # set ancestor index
            idx[-1] = ref_idx
        
        # save genealogy
        ancestor_idx[i-1] = idx
        
        
        
        ### Step 4: Calculate new weights (measurment update)
        sigma_y = Sigma_X[i,:,0,None]
        l_y = jax.vmap(
            functools.partial(log_likelihood_Normal, mean=Y[i], cov=R)
            )(sigma_y)
        weights[i] = np.exp(l_y - l_y_aux[idx])
        weights[i] = weights[i]/np.sum(weights[i])
        
        #abort
        if np.any(np.isnan(weights[i])):
            print("Particle degeneration at new weights")
            break
    
    
    ### Step 5: sample a new trajectory to return
    
    # draw trajectory index
    u = rng.random()
    idx_traj = np.searchsorted(np.cumsum(weights[i]), u)
    
    # reconstruct trajectory from genealogy
    x_traj = np.zeros((steps,2))
    x_traj[-1] = Sigma_X[-1, idx_traj]
    F_traj = np.zeros((steps,))
    F_traj[-1] = Sigma_F[-1, idx_traj]
    ancestry = np.zeros((steps,))
    ancestry[-1] = idx_traj
    for i in range(steps-2, -1, -1): # run backward in time
        ancestry[i] = ancestor_idx[i, int(ancestry[i+1])]
        x_traj[i] = Sigma_X[i, int(ancestry[i])]
        F_traj[i] = Sigma_F[i, int(ancestry[i])]
        
    
    return x_traj, F_traj