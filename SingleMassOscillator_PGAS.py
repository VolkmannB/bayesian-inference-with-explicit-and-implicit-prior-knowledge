import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools
from scipy.stats import invwishart


from src.SingleMassOscillator import F_spring, F_damper, f_x, f_y, N_ip, H, m
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, gaussian_RBF, prior_iw_calcStatistics, prior_niw_calcStatistics, prior_mniw_calcStatistics, prior_niw_2naturalPara_inv, prior_iw_2naturalPara_inv, prior_niw_2naturalPara
from src.Filtering import systematic_SISR, squared_error
from src.SingleMassOscillatorPlotting import generate_Animation






# function for conditional SMC
def conditional_SMC_with_ancestor_sampling(Y, U,                    # observations and inputs
                        P_niw_sample, F_mniw_sample, X_iw_sample,   # parameters
                        X_prop_im1,                                 # reference trajectory
                        f, h, basis_func, R,                        # model structures (including known covariance R)
                        N, dt):                                     # further settings
    # conditional_SMC_with_ancestor_sampling/PGAS Markov kernel according to Wigren.2022 (Algorithm S4) and Svensson.2017 (Algorithm 1)

    # setting initial variables
    T = len(X_prop_im1)

    # unpacking distribution parameters
    P_coeff = P_niw_sample[0]
    P_cov = P_niw_sample[1]

    F_coeff = F_mniw_sample[0]
    F_cov = F_mniw_sample[1]

    X_cov = X_iw_sample[0]

    # initialize probability distributions
    w_X = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), X_cov, n)
    w_F = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), F_cov, n)
    w_P = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), P_cov, n)



    # (1) - (2) initialize particles
    X = np.repeat(X_prop_im1[:,0],[1,N])
    X[:N-1,:] = X[:N-1,:] + w_X((N-1,2))

    phi_x0 = jax.vmap(basis_func)(X)
    #F = P_coeff @ phi_x0
    F = jax.vmap(jnp.matmul)(F_coeff, phi_x0) + w_F((N,))
    #F[:N-1] = F[:N-1] + w_F((N-1,))  # Is it necessary to have a noise-free reference for F and P as well?

    P = np.repeat(P_coeff,N,1) + w_P((N,))
    #P[:N-1] = P[:N-1] + w_P((N-1,))  # Is it necessary to have a noise-free reference for F and P as well?
    
    # store initial particles
    X_store = np.zeros((T,N,2))
    P_store = np.zeros((T,N))
    F_store = np.zeros((T,N))
    X_store[0,...] = X[None,...]
    P_store[0,...] = P[None,...]
    F_store[0,...] = F[None,...]



    # (3) loop
    for t in range(T-1):

        # (4) calculate first stage weights
        y_pred = jax.vmap(h)(X)
        w = jax.vmap(functools.partial(squared_error, y=Y[t], cov=R))(y_pred)
        w = w/np.sum(w)


        # (5) draw new indices
        u = np.random.rand()
        idx = np.array(systematic_SISR(u, w))
        idx[idx >= N] = N - 1 # correct out of bounds indices from numerical errors
        idx = idx[0:N-1]

        # (6) sample ancestor of the N-th particle
        #x_mean_tp1 = jax.vmap(functools.partial(f, F=U[t], dt=dt))(x=X, F_sd=F, p=P) 
        x_mean_tp1 = jax.vmap(functools.partial(f, F=U[t], dt=dt))(x=X[idx,:], F_sd=F[idx,:], p=P[idx,:]) 
        x_ref_tp1 = X_prop_im1[t+1,:]
        prob = w * jax.vmap(functools.partial(squared_error, y=x_ref_tp1, cov=X_cov))(x_mean_tp1)
        prob = prob/np.sum(prob)
        idx[N] = np.random.choice(range(0, N), size=1, p=prob)

        # (7) sample x for 1,...,N-1
        #X = jax.vmap(functools.partial(f, F=U[t], dt=dt))(x=X[idx,:], F_sd=F[idx,:], p=P[idx,:]) + w_X((N,2))
        X = x_mean_tp1 + w_X((N,2))
        
        # (8) sample x for N
        X[N,:] = x_ref_tp1

        # (9) sample F
        phi_x = jax.vmap(basis_func)(X)
        F = jax.vmap(jnp.matmul)(F_coeff, phi_x) + w_F((N,))

        # (9) sample P
        P = np.repeat(P_coeff,N,1) + w_P((N,))

        # (10) store trajectories for X
        X_store[0:t+1,...] = np.concatenate((X_store[0:t,idx,...], X[None,...]),axis=0)

        # (11) store trajectories for F
        F_store[0:t+1,...] = np.concatenate((F_store[0:t,idx,...], F[None,...]),axis=0)

        # (11) store trajectories for P
        P_store[0:t+1,...] = np.concatenate((P_store[0:t,idx,...], P[None,...]),axis=0)


    # (13) draw a sample trajectory as output
    sample = np.random.choice(range(0, N), size=1, p=w)
    X_prop_i = np.squeeze(X_store[:,sample,:]).T
    P_prop_i = np.squeeze(P_store[:,sample,:]).T
    F_prop_i = np.squeeze(F_store[:,sample,:]).T


    return X_prop_i, P_prop_i, F_prop_i





# function for parameter update
def parameter_update(X_prop_i, P_prop_i, F_prop_i,
                    X_prior, P_prior, F_prior,
                    f, basis_func, U,
                    dt, seed):
    
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
    P_cov = invwishart.rvs(
                df=P_df,
                scale=P_iw_scale,
                random_state=seed)
    
    P_cov_col = np.linalg.cholesky(P_cov/P_normal_scale)
    n_samples = np.random.normal(size=P_mean.shape)
    P_coeff = P_mean + np.matmul(P_cov_col,n_samples)
    #P_coeff = np.random.multivariate_normal(P_mean,P_cov/P_normal_scale), P_mean.shape

    P_niw_sample = list(P_coeff, P_cov)



    ### ---------------------------------------------------------
    ### xi 2 (F)
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    F_in = basis_func(X_prop_i)
    F_out = F_prop_i
    F_stats_new = list(prior_mniw_calcStatistics(*F_prior, F_in, F_out))

    # use suff. statistics to calculate posterior MNIW parameters
    F_mean, F_col_cov, F_iw_scale, F_df = prior_mniw_2naturalPara_inv(*F_stats_new)

    # covariance and mean update | state and xi 2 trajectories
    # sample from MNIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    F_row_cov = invwishart.rvs(
                df=F_df,
                scale=F_iw_scale,
                random_state=seed)
    
    
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
    X_in = f(X_prop_i[:,:-1], U, F_prop_i, dt=dt, p=P_prop_i)
    X_out = X_prop_i[:,1:]
    X_stats_new = prior_iw_calcStatistics(*X_prior, X_in, X_out)

    # use suff. statistics to calculate posterior IW parameters
    X_iw_scale, X_df = prior_iw_2naturalPara_inv(*X_stats_new)

    # covariance and mean update | state and xi 1 trajectories
    # sample from MNIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    X_cov = invwishart.rvs(
                df=X_df,
                scale=X_iw_scale,
                random_state=seed)
    
    X_iw_sample = list(X_cov)


    return X_iw_sample, P_niw_sample, F_mniw_sample





if __name__ == '__main__':

    ################################################################################
    # General settings

    N = 500    # no of particles
    t_end = 100.0
    dt = 0.01
    time = np.arange(0.0,t_end,dt)
    steps = len(time)
    seed = 275513 #np.random.randint(100, 1000000)
    rng = np.random.default_rng()
    print(f"Seed is: {seed}")
    np.random.seed(seed)



    ################################################################################
    # Generating offline training data

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
    X[0,...] = x0
    Y = np.zeros((steps,))
    F_sd = np.zeros((steps,))

    # input
    # F = np.ones((steps,)) * -9.81*m + np.sin(2*np.pi*np.arange(0,t_end,dt)/10) * 9.81
    U = np.zeros((steps,)) 
    U[int(t_end/(5*dt)):] = -9.81*m
    U[int(2*t_end/(5*dt)):] = -9.81*m*2
    U[int(3*t_end/(5*dt)):] = -9.81*m
    U[int(4*t_end/(5*dt)):] = 0


    # simulation loop
    for i in tqdm(range(1,steps), desc="Running simulation"):
        
        ####### Model simulation
        
        # update system state
        F_sd[i-1] = F_spring(X[i-1,0]) + F_damper(X[i-1,1])
        X[i] = f_x(X[i-1], U[i-1], F_sd[i-1], dt=dt) + w()
        
        # generate measurment
        Y[i] = X[i,0] + e()[0,0]







    ################################################################################
    # Offline identification

    # -----------------------------------------------------------------------------------
    ### Initialize first reference trajectory for X
    X_prop_im1 = np.zeros((2,steps)) 



    # -----------------------------------------------------------------------------------
    ### Initialize prior values for parameters

    # -----------------
    # X: parameters of the iw prior
    X_iw_scale = np.eye(2) # defining distribution parameters
    X_df = 2

    X_cov = invwishart.rvs(df=X_df, scale=X_iw_scale, random_state=seed) # sampling 
    X_iw_sample = list(X_cov)

    X_prior = list((X_iw_scale, X_df)) # storing natural parameters



    # -----------------
    # P: parameters of the niw prior
    P_mean = np.eye(1) # defining distribution parameters
    P_normal_scale = np.eye(1)
    P_iw_scale = np.eye(1)
    P_df = 1

    P_cov = invwishart.rvs(df=P_df, scale=P_iw_scale, random_state=seed) # sampling 
    P_cov_col = np.linalg.cholesky(P_cov/P_normal_scale)
    n_samples = np.random.normal(size=P_mean.shape)
    P_coeff = P_mean + np.matmul(P_cov_col,n_samples)
    P_niw_sample = list((P_coeff, P_cov))

    eta_0, eta_1, eta_2, eta_3  = prior_niw_2naturalPara(P_mean, P_normal_scale, P_iw_scale, P_df) # storing natural parameters
    P_prior = list((eta_0, eta_1, eta_2, eta_3))



    # -----------------
    # F: parameters of the mniw prior
    F_mean = np.zeros((1, N_ip)) # defining distribution parameters
    F_col_cov = np.eye(N_ip)*40
    F_iw_scale = np.eye(1)
    F_df = 1

    F_row_cov = invwishart.rvs(df=F_df, scale=F_iw_scale, random_state=seed) # sampling 
    F_row_cov_col = np.sqrt(F_row_cov)
    F_col_cov_col = np.linalg.cholesky(F_col_cov)
    n_samples = np.random.normal(size=F_mean.shape)
    F_coeff = F_mean + F_row_cov_col*np.matmul(n_samples,F_col_cov_col)
    F_mniw_sample = list((F_coeff, F_row_cov))

    eta_1, eta_2, eta_0, eta_3 = prior_mniw_2naturalPara(F_mean, F_col_cov, F_iw_scale, F_df) # storing natural parameters
    F_prior = list((eta_0, eta_1, eta_2, eta_3))






    # -----------------------------------------------------------------------------------
    ### identification loop
    for i in tqdm(range(1,steps), desc="Running identification"):
        
        ### Step 1: Get new state trajectory proposal from PGAS Markov kernel with bootstrap PF | Given the parameters
        X_prop_i, P_prop_i, F_prop_i = conditional_SMC_with_ancestor_sampling(Y, U,                     # observations and inputs
                                                            P_niw_sample, F_mniw_sample, X_iw_sample,   # parameters
                                                            X_prop_im1,                                 # reference trajectory
                                                            f_x, f_y, H, R,                             # model structures (including known covariance R)
                                                            N, dt)                                      # further settings

        ### Step 2: Compute parameter posterior | Given the state trajectory
        P_niw_sample, F_mniw_sample, X_iw_sample = parameter_update(X_prop_i, P_prop_i, F_prop_i,       # samples for the latent variable sequences
                                                                X_prior, P_prior, F_prior,              # initial values for parameter priors
                                                                f_x, H, U,                              # model structures (including known covariance R)
                                                                dt, seed)                               # further settings



    # ToDo: 
    # 1. Animation plots


    ################################################################################
    # Plots

    # fig = generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, H, W, CW, time, 200., 30., 30)

    # print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-Sigma_F)))

    # fig.show()