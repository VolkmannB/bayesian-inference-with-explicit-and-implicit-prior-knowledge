import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.SingleMassOscillator import F_spring, F_damper, f_x, N_ip, ip, H, m
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics, prior_mniw_sampleLikelihood, gaussian_RBF, prior_iw_updateStatistics
from src.Filtering import systematic_SISR, squared_error
from src.SingleMassOscillatorPlotting import generate_Animation



################################################################################
# General settings

N = 1500    # no of particles
t_end = 100.0
dt = 0.01
time = np.arange(0.0,t_end,dt)
steps = len(time)
seed = 275513 #np.random.randint(100, 1000000)
rng = np.random.default_rng()
print(f"Seed is: {seed}")
np.random.seed(seed)
key = jax.random.key(np.random.randint(100, 1000))
print(f"Initial jax-key is: {key}")



################################################################################
# Generating offline training data

# Question: is there no process noise?!


# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])


# noise
R = np.array([[1e-3]])
Q = np.diag([5e-6, 5e-7])
w = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), R, n)


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


# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    
    # update system state
    F_sd[i-1] = F_spring(X[i-1,0]) + F_damper(X[i-1,1])
    X[i] = f_x(X[i-1], F[i-1], F_sd[i-1], dt=dt)
    
    # generate measurment
    Y[i] = X[i,0] + e()[0,0]







################################################################################
# Offline identification

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




# ToDo: 
# 1. Initialize parameter priors and trajectories
# 2. Implement functions to update and translate iw & niw distributions
# 3. Adjust distribution functions to return samples of the posterior ((m)(n)iw) instead of the predictive ((m)t) distribution!
# 4. Understand Jax notation and change functions/notation accordingly
# 5. How to calculate the suff. statistics of the physical parameters? --> Derive!
# 6. Animation plots
# 7. Implement PGAS Markov kernel further








# function for Markov kernel
def PGAS_Markov_kernel(X_prop_im1, P_prop_im1, F_prop_im1, 
                        P_niw_sample, F_mniw_sample, X_iw_sample, 
                        N, f_x, H, R, dt, Y, U):

    # unpacking distribution parameters
    P_mu = P_niw_sample[0]
    P_Q = P_niw_sample[1]

    F_M = F_mniw_sample[0]
    F_Q = F_mniw_sample[1]

    X_Q = X_iw_sample[0]

    # initialize particles
    w_x = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), X_Q, n)
    X[0,...] = X_prop_im1[0] + w_x((N-1,))
    X[0,N-1] = X_prop_im1[0]

    weights = np.ones((steps,N))/N

    # w_x = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), X_Q, n)
    # x = np.repeat(X_prop_im1[0], N, 0) + w_x((N,))
    # x[0,N-1] = X_prop_im1[0]


    for t in range(len(X_prop_im1)):

        # calculate first stage weights
        l = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(X[:,0,None])
        l = l/np.sum(l)

        # draw new indices
        u = np.random.rand()
        idx = np.array(systematic_SISR(u, l))
        idx[idx >= N] = N - 1 # correct out of bounds indices from numerical errors
        X = X[idx,...]

        # sample from proposal for x, p, and F at time t

        # continue here...




    return X_prop_i, P_prop_i, F_prop_i





# function for parameter update
def parameter_update(X_prop_i, P_prop_i, F_prop_i, 
                    P_stats, F_stats, X_stats, 
                    f_x, H):
    
    ### ---------------------------------------------------------
    ### xi 1 (P)
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    P_in = ???
    P_out = P_prop_i
    P_stats_new = prior_mniw_updateStatistics(P_stats, P_in, P_out)

    # use suff. statistics to calculate posterior NIW parameters
    P_niw_para = list(jax.vmap(prior_mniw_2naturalPara_inv)(P_stats_new))

    # covariance and mean update | state and xi 1 trajectories
    # sample from NIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    key, *keys = jax.random.split(key, N+1)
    P_niw_sample = jax.vmap(prior_niw_sampleLikelihood)(
        key=jnp.asarray(keys),
        mu=P_niw_para[0],
        kappa=P_niw_para[1],
        Psi=P_niw_para[2],
        nu=P_niw_para[3]
    ).flatten()



    ### ---------------------------------------------------------
    ### xi 2 (F)
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    F_in = jax.vmap(H)(X_prop_i)
    F_out = F_prop_i
    F_stats_new = prior_mniw_updateStatistics(F_stats, F_in, F_out)

    # use suff. statistics to calculate posterior MNIW parameters
    F_mniw_para = list(jax.vmap(prior_mniw_2naturalPara_inv)(F_stats_new))

    # covariance and mean update | state and xi 2 trajectories
    # sample from MNIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    key, *keys = jax.random.split(key, N+1)
    F_mniw_sample = jax.vmap(prior_mniw_sampleLikelihood)(
        key=jnp.asarray(keys),
        M=F_mniw_para[0],
        V=F_mniw_para[1],
        Psi=F_mniw_para[2],
        nu=F_mniw_para[3]
    ).flatten()



    ### ---------------------------------------------------------
    ### states (X), overall covariance update | trajectories of states and xis 
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    w_x = w((N,))
    X_in = jax.vmap(
        functools.partial(f_x, F=F[i-1], dt=dt)
        )(
            x       =X_prop_i, 
            F_sd    =F_prop_i,
            P       =P_prop_i
            ) + w_x
    X_out = X_prop_i
    X_stats_new = prior_mniw_updateStatistics(X_stats, X_in, X_out)

    # use suff. statistics to calculate posterior IW parameters
    X_iw_para = list(jax.vmap(prior_iw_2naturalPara_inv)(X_stats_new))

    # covariance and mean update | state and xi 1 trajectories
    # sample from MNIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    key, *keys = jax.random.split(key, N+1)
    X_iw_sample = jax.vmap(prior_iw_sampleLikelihood)(
        key=jnp.asarray(keys),
        Psi=X_iw_para[0],
        nu=X_iw_para[1]
    ).flatten()


    return P_niw_sample, F_mniw_sample, X_iw_sample






# identification loop
for i in tqdm(range(1,steps), desc="Running identification"):
    
    ### Step 1: Get new state trajectory proposal from PGAS Markov kernel with bootstrap PF | Given the parameters
    X_prop_i, P_prop_i, F_prop_i = PGAS_Markov_kernel(X_prop_im1, P_prop_im1, F_prop_im1, 
                                                        P_niw_sample, F_mniw_sample, X_iw_sample, 
                                                        N, f_x, H, R, dt, Y, U)

    ### Step 2: Compute parameter posterior | Given the state trajectory
    P_niw_sample, F_mniw_sample, X_iw_sample = parameter_update(X_prop_i, P_prop_i, F_prop_i, 
                                                P_stats_prior, F_stats_prior, X_stats_prior, 
                                                f_x, H)





################################################################################
# Plots

# fig = generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, H, W, CW, time, 200., 30., 30)

# print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-Sigma_F)))

# fig.show()