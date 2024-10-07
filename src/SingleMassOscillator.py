import numpy as np
import jax.numpy as jnp
import jax
import scipy.signal
from tqdm import tqdm
import functools
import scipy

from src.BayesianInferrence import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_calcStatistics
from src.BayesianInferrence import prior_mniw_Predictive
from src.BayesianInferrence import prior_mniw_log_base_measure
from src.Filtering import systematic_SISR, log_likelihood_Normal
from src.Filtering import reconstruct_trajectory


#### This section defines the state space model

# parameters
m=2.0
c1=10.0
c2=2.0
d1=0.8
d2=0.4
C = np.array([[1,0]])


def F_spring(x):
    return c1 * x + c2 * x**3



def F_damper(dx):
    return d1*dx * (1/(1+d2*dx*jnp.tanh(dx)))



def dx(x, F, F_sd, m=m):
    return jnp.array(
        [x[1], -F_sd/m + F/m]
    )



@jax.jit
def f_x(x, F, F_sd, dt):
    
    # Runge-Kutta 4
    k1 = dx(x, F, F_sd)
    k2 = dx(x+dt/2.0*k1, F, F_sd)
    k3 = dx(x+dt/2.0*k2, F, F_sd)
    k4 = dx(x+dt*k3, F, F_sd)
    x = x + dt/6.0*(k1+2*k2+2*k3+k4) 
    
    return x



@jax.jit
def f_y(x):
    return C @ x



#### This section defines relevant parameters for the simulation

# set a seed for reproducability
rng = np.random.default_rng(16723573)

# simulation parameters
N_particles = 200
N_PGAS_iter = 5
t_end = 50.0
dt = 0.02
forget_factor = 0.999
time = np.arange(0.0,t_end,dt)
steps = len(time)

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
F_ext = np.ones((steps,)) * 9.81*m
F_ext[int(t_end/(5*dt)):] = 0
F_ext[int(2*t_end/(5*dt)):] = -9.81*m
F_ext[int(3*t_end/(5*dt)):] = 0
F_ext[int(4*t_end/(5*dt)):] = 9.81*m
    
    
    
#### This section defines the basis function expansion

# basis functions
N_basis_fcn = 41
basis_fcn, sd = generate_Hilbert_BasisFunction(
    num_fcn=N_basis_fcn, 
    domain_boundary=np.array([[-7.5, 7.5],[-7.5, 7.5]]), 
    lengthscale=7.5*2/N_basis_fcn, 
    scale=50
    )



# parameters of the MNIW prior
GP_prior = list(prior_mniw_2naturalPara(
    np.zeros((1, N_basis_fcn)),
    np.diag(sd),
    np.eye(1)*8,
    1
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
    Sigma_X = np.zeros((steps, N_particles, 2))
    Sigma_F = np.zeros((steps, N_particles))
    log_weights = np.ones((steps, N_particles))/N_particles
    
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
    weights = np.exp(log_weights[0] - np.max(log_weights[0]))
    weights /= np.sum(weights)
    GP_stats_logging[0][0] = np.einsum('n...,n->...', GP_stats[0], weights)
    GP_stats_logging[1][0] = np.einsum('n...,n->...', GP_stats[1], weights)
    GP_stats_logging[2][0] = np.einsum('n...,n->...', GP_stats[2], weights)
    GP_stats_logging[3][0] = np.einsum('n...,n->...', GP_stats[3], weights)
    
    
    
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
        log_weights_aux = log_weights[i-1] + l_y_aux
        weights_aux = np.exp(log_weights_aux - np.max(log_weights_aux))
        weights_aux /= np.sum(weights_aux)
        
        #abort
        if np.any(np.isnan(weights_aux)):
            print("Particle degeneration at auxiliary weights")
            break
        
        # draw new indices
        u = rng.random()
        idx = np.array(systematic_SISR(u, weights_aux))
        # correct out of bounds indices from numerical errors
        idx[idx >= N_particles] = N_particles - 1 
        
        
        
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
        basis = jax.vmap(basis_fcn)(Sigma_X[i])
        
        # calculate conditional predictive distribution
        c_mean, c_col_scale, c_row_scale, df = jax.vmap(
            functools.partial(prior_mniw_Predictive)
            )(
                mean=GP_para[0][idx],
                col_cov=GP_para[1][idx],
                row_scale=GP_para[2][idx],
                df=GP_para[3][idx],
                basis=basis
        )
        
        # generate samples
        c_col_scale_chol = np.sqrt(np.squeeze(c_col_scale))
        c_row_scale_chol = np.sqrt(np.squeeze(c_row_scale))
        t_samples = rng.standard_t(df=df)
        Sigma_F[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
        
        
        
        ### Step 4: Update the sufficient statistics of GP with new proposal
        T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_F[i],
            basis
        )
        GP_stats[0] = GP_stats[0][idx] + T_0
        GP_stats[1] = GP_stats[1][idx] + T_1
        GP_stats[2] = GP_stats[2][idx] + T_2
        GP_stats[3] = GP_stats[3][idx] + T_3
        
        # calculate new weights (measurment update)
        sigma_y = Sigma_X[i,:,0,None]
        l_y = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(sigma_y)
        log_weights[i] = l_y - l_y_aux[idx]
        
        
        
        # logging
        weights = np.exp(log_weights[i] - np.max(log_weights[i]))
        weights /= np.sum(weights)
        GP_stats_logging[0][i] = np.einsum('n...,n->...', GP_stats[0], weights)
        GP_stats_logging[1][i] = np.einsum('n...,n->...', GP_stats[1], weights)
        GP_stats_logging[2][i] = np.einsum('n...,n->...', GP_stats[2], weights)
        GP_stats_logging[3][i] = np.einsum('n...,n->...', GP_stats[3], weights)
        
        #abort
        if np.any(np.isnan(log_weights[i])):
            print("Particle degeneration at new weights")
            break
    
    
    
    # normalize weights
    log_weights -= np.max(log_weights, axis=-1, keepdims=True)
    weights = np.exp(log_weights)
    weights /= np.sum(weights, axis=-1, keepdims=True)
        
    return Sigma_X, Sigma_F, weights, GP_stats_logging



#### This section defines a function to run the offline version of the algorithm

def SingleMassOscillator_PGAS(Y):
    """
    This function runs the PGAS algorithm.

    Args:
        Y (Array): Measurements
    """
    
    Sigma_X = np.zeros((steps,N_PGAS_iter,2))
    Sigma_F = np.zeros((steps,N_PGAS_iter))
    weights = np.ones((steps,N_PGAS_iter))/N_PGAS_iter
    
    # variable for sufficient statistics
    GP_stats = [
        np.zeros((N_PGAS_iter, N_basis_fcn, 1)),
        np.zeros((N_PGAS_iter, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_PGAS_iter, 1, 1)),
        np.zeros((N_PGAS_iter,))         # nu > n_xi +1
    ]

    # set initial reference using filtering distribution from APF
    print(f"Setting initial reference trajectory")
    init_X, init_F, init_w, _ = SingleMassOscillator_APF(Y)
    idx = np.searchsorted(np.cumsum(init_w[-1]), rng.random())
    Sigma_X[:,0] = init_X[:,idx]
    Sigma_F[:,0] = init_F[:,idx]
        
        
    # make proposal for distribution of F_sd using new proposals of trajectories
    basis = jax.vmap(basis_fcn)(Sigma_X[:,0])
    T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
        Sigma_F[:,0],
        basis
    )
    GP_stats[0][0] = np.sum(T_0, axis=0)
    GP_stats[1][0] = np.sum(T_1, axis=0)
    GP_stats[2][0] = np.sum(T_2, axis=0)
    GP_stats[3][0] = np.sum(T_3, axis=0)



    ### Run PGAS
    for k in range(1, N_PGAS_iter):
        print(f"Starting iteration {k}")
        
        # sample new proposal for trajectories using CPF with AS
        Sigma_X[:,k], Sigma_F[:,k] = SingleMassOscillator_CPFAS_Kernel(
            Y=Y,
            x_ref=Sigma_X[:,k-1],
            F_ref=Sigma_F[:,k-1],
            GP_stats_ref=[
                GP_stats[0][k-1],
                GP_stats[1][k-1],
                GP_stats[2][k-1],
                GP_stats[3][k-1]
                ]
        )
        
        
        # make proposal for distribution of F_sd using new proposals of trajectories
        basis = jax.vmap(basis_fcn)(Sigma_X[:,k])
        T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_F[:,k],
            basis
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
    GP_stats_ref
    ):
    """
    This function runs the CPFAS kernel for the PGAS algorithm to generate new 
    proposals for the state trajectory. It is a conditional auxiliary particle 
    filter. 
    
    """
    
    # Particle trajectories
    Sigma_X = np.zeros((steps,N_particles,2))
    Sigma_F = np.zeros((steps,N_particles))
    log_weights = np.zeros((steps,N_particles))
    ancestor_idx = np.zeros((steps-1,N_particles))
    
    ## set initial values
    Sigma_X[0,...] = rng.multivariate_normal(x0, P0, (N_particles,)) 
    Sigma_F[0,...] = rng.normal(0, P0_F, (N_particles,))
    Sigma_X[0,-1] = x_ref[0]
    Sigma_F[0,-1] = F_ref[0]
    
    ## split model into reference and ancestor statistics
    
    # calculate ancestor statistics
    basis = jax.vmap(basis_fcn)(Sigma_X[0,...])
    GP_stats_ancestor = list(jax.vmap(prior_mniw_calcStatistics)(
        Sigma_F[0,...],
        basis
    ))
    
    # update reference statistic
    basis = basis_fcn(x_ref[0])
    T_0, T_1, T_2, T_3 = prior_mniw_calcStatistics(F_ref[0], basis)
    GP_stats_ref[0] -= T_0
    GP_stats_ref[1] -= T_1
    GP_stats_ref[2] -= T_2
    GP_stats_ref[3] -= T_3
    
    
    
    for i in tqdm(range(1,steps), desc="    Running CPF Kernel"):
        
        # calculate models
        Mean, Col_Cov, Row_Scale, df = jax.vmap(prior_mniw_2naturalPara_inv)(
            GP_prior[0] + GP_stats_ref[0] + GP_stats_ancestor[0],
            GP_prior[1] + GP_stats_ref[1] + GP_stats_ancestor[1],
            GP_prior[2] + GP_stats_ref[2] + GP_stats_ancestor[2],
            GP_prior[3] + GP_stats_ref[3] + GP_stats_ancestor[3]
        )
        
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
        log_weights_aux = log_weights[i-1] + l_y_aux
        weights_aux = np.exp(log_weights_aux - np.max(log_weights_aux))
        weights_aux /= np.sum(weights_aux)
        
        #abort
        if np.any(np.isnan(weights_aux)):
            raise ValueError("Particle degeneration at auxiliary weights")
        
        # draw new indices
        idx = np.array(systematic_SISR(rng.random(), weights_aux))
        
        
        
        ### Step 2: Sample a new ancestor for the reference trajectory

        g_T = jax.vmap(
            prior_mniw_log_base_measure
            )(
            GP_prior[0] + GP_stats_ancestor[0] + GP_stats_ref[0],
            GP_prior[1] + GP_stats_ancestor[1] + GP_stats_ref[1],
            GP_prior[2] + GP_stats_ancestor[2] + GP_stats_ref[2],
            GP_prior[3] + GP_stats_ancestor[3] + GP_stats_ref[3]
        )
        g_t = jax.vmap(
            prior_mniw_log_base_measure
            )(
            GP_prior[0] + GP_stats_ancestor[0],
            GP_prior[1] + GP_stats_ancestor[1],
            GP_prior[2] + GP_stats_ancestor[2],
            GP_prior[3] + GP_stats_ancestor[3]
        )

        # calculate ancestor weights
        h_x = jax.vmap(
            functools.partial(
            log_likelihood_Normal, 
            observed=x_ref[i], 
            cov=Q)
            )(
            mean=x_aux, 
        )
        

        log_weights_ancestor = log_weights_aux + g_t - g_T + h_x
        weights_ancestor = np.exp(log_weights_ancestor - np.max(log_weights_ancestor))
        weights_ancestor /= np.sum(weights_ancestor)
        
        # sample an ancestor index for reference trajectory
        ref_idx = np.searchsorted(np.cumsum(weights_ancestor), rng.random())
        
        # set ancestor index
        idx[-1] = ref_idx
        
        # correct out of bounds indices from numerical errors
        idx[idx >= N_particles] = N_particles - 1
        
        # save genealogy
        ancestor_idx[i-1] = idx
        
        
        
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
        
        # set reference trajectory for state x
        Sigma_X[i,-1] = x_ref[i]
        
        ## sample from proposal for F at time t
        # evaluate basis functions for all particles
        basis = jax.vmap(basis_fcn)(Sigma_X[i])
        
        # calculate predictive distribution
        c_mean, c_col_scale, c_row_scale, df = jax.vmap(
            functools.partial(prior_mniw_Predictive)
            )(
            basis=basis, 
            mean=Mean[idx],
            col_cov=Col_Cov[idx],
            row_scale=Row_Scale[idx],
            df=df[idx]
        )
        
        # generate samples
        c_col_scale_chol = np.squeeze(np.linalg.cholesky(c_col_scale))
        c_row_scale_chol = np.squeeze(np.linalg.cholesky(c_row_scale))
        t_samples = rng.standard_t(df=df)
        Sigma_F[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
        
        # set reference trajectory for F_sd
        Sigma_F[i,-1] = F_ref[i]
        
        
        
        ### Step 4: Update reference statistic
        basis_ref = basis_fcn(x_ref[i])
        T_0, T_1, T_2, T_3 = prior_mniw_calcStatistics(
            F_ref[i],
            basis_ref
        )
        GP_stats_ref[0] -= T_0
        GP_stats_ref[1] -= T_1
        GP_stats_ref[2] -= T_2
        GP_stats_ref[3] -= T_3
        
        
        
        ### Step 5: Update hyperparameters
        T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_F[i],
            basis
        )
        GP_stats_ancestor[0] = GP_stats_ancestor[0][idx] + T_0
        GP_stats_ancestor[1] = GP_stats_ancestor[1][idx] + T_1
        GP_stats_ancestor[2] = GP_stats_ancestor[2][idx] + T_2
        GP_stats_ancestor[3] = GP_stats_ancestor[3][idx] + T_3
        
        
        
        ### Step 6: Calculate new weights (measurment update)
        sigma_y = Sigma_X[i,:,0,None]
        l_y = jax.vmap(
            functools.partial(log_likelihood_Normal, mean=Y[i], cov=R)
            )(sigma_y)
        log_weights[i] = l_y - l_y_aux[idx]
        
        #abort
        if np.any(np.isnan(log_weights[i])):
            print("Particle degeneration at new weights")
            break
    
    
    ### Step 7: sample a new trajectory to return
    
    # draw trajectory index
    weights = np.exp(log_weights[-1] - np.max(log_weights[-1]))
    weights /= np.sum(weights)
    idx_traj = np.searchsorted(np.cumsum(weights), rng.random())
    
    # reconstruct trajectory from genealogy
    x_traj = reconstruct_trajectory(Sigma_X, ancestor_idx, idx_traj)
    F_traj = reconstruct_trajectory(Sigma_F, ancestor_idx, idx_traj)
        
    
    return x_traj, F_traj