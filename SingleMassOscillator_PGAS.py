import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools
from scipy.stats import invwishart


from src.SingleMassOscillator import F_spring, F_damper, f_x, N_ip, ip, H, m
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics, prior_mniw_sampleLikelihood, gaussian_RBF, prior_iw_calcStatistics, prior_niw_calcStatistics, prior_mniw_calcStatistics, prior_niw_2naturalPara_inv, prior_iw_2naturalPara_inv
from src.Filtering import systematic_SISR, squared_error
from src.SingleMassOscillatorPlotting import generate_Animation






# function for conditional SMC
def conditional_SMC_with_ancestor_sampling(Y, U,                    # observations and inputs
                        P_niw_sample, F_mniw_sample, X_iw_sample,   # parameters
                        X_prop_im1,                                 # reference trajectory
                        N, dt,                                      # further settings
                        f, h, basis_func, R):                       # model structures (including known covariance R)
    # conditional_SMC_with_ancestor_sampling/PGAS Markov kernel according to Wigren.2022 (Algorithm S4) and Svensson.2017 (Algorithm 1)

    # setting initial variables
    T = len(X_prop_im1)


    # unpacking distribution parameters
    P_m = P_niw_sample[0]
    P_Q = P_niw_sample[1]

    F_M = F_mniw_sample[0]
    F_Q = F_mniw_sample[1]

    X_Q = X_iw_sample[0]

    # (1) - (2) initialize particles
    w_X = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), X_Q, n)
    w_F = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), F_Q, n)
    w_P = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), P_Q, n)

    X = X_prop_im1[0,...] + w_X((N,))
    X[N-1,...] = X_prop_im1[0,...]

    X_store = np.zeros((T,N,2))
    P_store = np.zeros((T,N))
    F_store = np.zeros((T,N))


    # w_x = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), X_Q, n)
    # x = np.repeat(X_prop_im1[0], N, 0) + w_x((N,))
    # x[0,N-1] = X_prop_im1[0]


    
    for t in range(T):

        # (4) calculate first stage weights
        y_pred = h(X[:,:,None])
        l = jax.vmap(functools.partial(squared_error, y=Y[t], cov=R))(y_pred)
        l = l/np.sum(l)


        # draw new indices
        u = np.random.rand()
        idx = np.array(systematic_SISR(u, l))
        idx[idx >= N] = N - 1 # correct out of bounds indices from numerical errors
        idx = idx[0:N-1]
        X = X[:,idx,...]
        l = l[idx]


        # sampling from proposal
        # for P at time t+1
        P = np.random.multivariate_normal(P_mu, P_Q, (N,)) 

        # for F at time t+1
        phi = jax.vmap(basis_func)(X[t,:N-1])
        xi = np.matmul(F_M,phi)
        F =  np.random.multivariate_normal(xi, F_Q, (N,)) 

        # for x at time t+1
        X_tp1_mean = jax.vmap(
            functools.partial(f, F=U[t], dt=dt)
            )(
                x=X[t,:N-1,:], 
                F_sd=F[t,:N-1],
                P = P[t,:N-1,:]
                ) 
        X[t+1,:N-1,:] = X_tp1_mean + np.random.multivariate_normal(np.zeros((2,)), X_Q, (N,)) 


        # set x_t+1^N to input trajectory (for ancestor sampling)
        X[t+1,N-1:N,:] = X_prop_im1[t+1,:]


        # sample ancestor of the N-1 -th particles
        prob = l * jax.vmap(functools.partial(squared_error, y=X[t+1,N-1,:], cov=X_Q))(X_tp1_mean)
        prob = prob/np.sum(prob)
        idx[N-1:N] = np.random.choice(range(0, N), size=1, p=prob)


        # store trajectories
        X_store[i,...] = X[None,...]
        P_store[i,...] = P[None,...]
        F_store[i,...] = F[None,...]

    # draw a sample trajectory as output
    sample = np.random.choice(range(0, N), size=1, p=l)
    X_prop_i = X_store[:,sample,:]
    P_prop_i = P_store[:,sample,:]
    F_prop_i = F_store[:,sample,:]


    return X_prop_i, P_prop_i, F_prop_i





# function for parameter update
def parameter_update(X_prop_i, P_prop_i, F_prop_i,
                    X_prior, P_prior, F_prior,
                    f_x, H, seed):
    
    ### ---------------------------------------------------------
    ### xi 1 (P)
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    P_out = P_prop_i
    P_stats_new = list(prior_niw_calcStatistics(*P_prior, P_out))

    # use suff. statistics to calculate posterior NIW parameters
    P_mean, P_normal_scale, P_iw_scale, P_df = prior_niw_2naturalPara_inv(*P_stats_new)

    # covariance and mean update | state and xi 1 trajectories
    # sample from NIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    P_cov = invwishart(
                df=P_df,
                scale=P_iw_scale,
                seed=seed)
    
    P_cov_col = np.linalg.cholesky(P_cov/P_normal_scale)
    n_samples = np.random.normal(size=P_mean.shape)
    P_coeff = P_mean + np.matmul(P_cov_col,n_samples)
    #P_coeff = np.random.multivariate_normal(P_mean,P_cov/P_normal_scale), P_mean.shape

    P_niw_sample = list(P_coeff, P_cov)



    ### ---------------------------------------------------------
    ### xi 2 (F)
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    F_in = H(X_prop_i)
    F_out = F_prop_i
    F_stats_new = list(prior_mniw_calcStatistics(*F_prior, F_in, F_out))

    # use suff. statistics to calculate posterior MNIW parameters
    F_mean, F_col_cov, F_iw_scale, F_df = prior_mniw_2naturalPara_inv(*F_stats_new)

    # covariance and mean update | state and xi 2 trajectories
    # sample from MNIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    F_row_cov = invwishart(
                df=F_df,
                scale=F_iw_scale,
                seed=seed)
    
    
    F_row_cov_col = np.linalg.cholesky(F_row_cov)
    F_col_cov_col = np.linalg.cholesky(F_col_cov)
    n_samples = np.random.normal(size=F_mean.shape)
    F_coeff = F_mean + np.matmul(F_row_cov_col,np.matmul(n_samples,F_col_cov_col))
    # F_cov = np.kron(F_col_cov, F_row_cov)
    # F_coeff = np.reshape(np.random.multivariate_normal(F_mean[:],F_cov), F_mean.shape)

    F_mniw_sample = list(F_coeff, F_row_cov)



    ### ---------------------------------------------------------
    ### states (X), overall covariance update | trajectories of states and xis 
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    X_in = f_x(X_prop_i[:,:-1], F, F_prop_i, dt=dt, m=P_prop_i)
    X_out = X_prop_i[:,1:]
    X_stats_new = prior_iw_calcStatistics(*X_prior, X_in, X_out)

    # use suff. statistics to calculate posterior IW parameters
    X_iw_scale, X_df = prior_iw_2naturalPara_inv(*X_stats_new)

    # covariance and mean update | state and xi 1 trajectories
    # sample from MNIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    X_cov = invwishart(
                df=X_df,
                scale=X_iw_scale,
                seed=seed)
    
    X_iw_sample = list(X_cov)


    return X_iw_sample, P_niw_sample, F_mniw_sample





if __name__ == '__main__':

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




    # identification loop
    for i in tqdm(range(1,steps), desc="Running identification"):
        
        ### Step 1: Get new state trajectory proposal from PGAS Markov kernel with bootstrap PF | Given the parameters
        X_prop_i, P_prop_i, F_prop_i = conditional_SMC_with_ancestor_sampling(X_prop_im1, P_prop_im1, F_prop_im1, 
                                                            P_niw_sample, F_mniw_sample, X_iw_sample, 
                                                            N, f_x, H, R, dt, Y, U)

        ### Step 2: Compute parameter posterior | Given the state trajectory
        P_niw_sample, F_mniw_sample, X_iw_sample = parameter_update(X_prop_i, P_prop_i, F_prop_i, 
                                                    X_prior, P_prior, F_prior,
                                                    f_x, H)





    ################################################################################
    # Plots

    # fig = generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, H, W, CW, time, 200., 30., 30)

    # print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-Sigma_F)))

    # fig.show()