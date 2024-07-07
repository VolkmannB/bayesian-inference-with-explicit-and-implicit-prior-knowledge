import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools
from scipy.stats import invwishart

from src.conditional_smc import CSMC
from src.utils import multinomial_resampling, systematic_resampling, stratified_resampling


from src.SingleMassOscillator import F_spring, F_damper, f_x, f_y, N_ip, basis_fcn, sd, m
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, gaussian_RBF, prior_iw_calcStatistics, prior_niw_calcStatistics, prior_mniw_calcStatistics, prior_niw_2naturalPara_inv, prior_iw_2naturalPara_inv, prior_niw_2naturalPara
from src.Filtering import systematic_SISR, squared_error, logweighting
from src.SingleMassOscillatorPlotting import generate_Animation







# function for conditional SMC
def conditional_SMC_with_ancestor_sampling(Y, U,                    # observations and inputs
                        X_iw_sample, P_niw_sample, F_mniw_sample,   # parameters
                        x0, X_prop_im1,                                 # reference trajectory
                        f_x, f_y, basis_func, R,                        # model structures (including known covariance R)
                        n_particles, dt, X_true, P_true, F_true, i, ancestor_sampling = True):                                     # further settings
    # conditional_SMC_with_ancestor_sampling/PGAS Markov kernel according to Wigren.2022 (Algorithm S4) and Svensson.2017 (Algorithm 1)

    # setting sequence length
    seq_len = X_prop_im1.shape[1]

    # unpacking distribution parameters
    P_coeff = P_niw_sample[0]
    P_cov = P_niw_sample[1]

    F_coeff = F_mniw_sample[0]
    F_cov = F_mniw_sample[1]

    X_cov = X_iw_sample[0]

    # initialize probability distributions
    w_X = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), X_cov, n)
    w_F = lambda n=1: np.random.normal(0, np.squeeze(F_cov), n)
    w_P = lambda n=1: np.random.normal(0, np.squeeze(P_cov), n)



    # (1) - (2) initialize particles
    X_particles = np.zeros((n_particles,2)) # filter
    X_particles[...] = x0 # deterministic initial condition
    X_particles[n_particles-1,:] = X_prop_im1[:,0] # reference trajectory

    phi_x0 = jax.vmap(basis_func)(X_particles)
    #F_particles = P_coeff @ phi_x0
    F_particles = np.squeeze(jax.vmap(lambda x: jnp.matmul(F_coeff,x))(phi_x0))
    #F_particles[:n_particles-1] = F_particles[:n_particles-1] + w_F((n_particles-1,))  # Is it necessary to have a noise-free reference for F_particles and P_particles as well?

    P_particles = np.ones(n_particles)*np.squeeze(P_coeff)
    #P_particles[:n_particles-1] = P_particles[:n_particles-1] + w_P((n_particles-1,))  # Is it necessary to have a noise-free reference for F_particles and P_particles as well?
    
    # store initial particles
    X_store = np.zeros((seq_len,n_particles,2))
    P_store = np.zeros((seq_len,n_particles))
    F_store = np.zeros((seq_len,n_particles))
    X_store[0,...] = X_particles[None,...]
    P_store[0,...] = P_particles[None,...]
    F_store[0,...] = F_particles[None,...]


    # (4) calculate first stage weights
    y_pred = jax.vmap(f_y)(X_particles)
    # w = jax.vmap(functools.partial(squared_error, y=Y[t], cov=R))(y_pred)
    # w = np.asarray(w).astype('float64')
    # w = w/np.sum(w)

    logweights = jax.vmap(functools.partial(logweighting, y=Y[0], cov=R))(y_pred)
    # logweights = -(1 / (2 * R)) * (Y[t] - y_pred) ** 2
    max_weight = max(logweights)  # Subtract the maximum value for numerical stability
    w = np.exp(logweights - max_weight)
    w = np.asarray(w).astype('float64')
    w = np.squeeze(w / sum(w))  # Save the normalized weights
    # accumulate the log-likelihood
    # log_likelihood += max_weight + np.log(sum(w)) - np.log(n_particles)

    # (3) loop
    log_likelihood = 0
    weight_depletion = False
    for t in range(seq_len-1):

        # (5) draw new indices
        # u = np.random.rand()
        # idx = np.array(systematic_SISR(u, w))
        # idx[idx >= n_particles] = n_particles - 1 # correct out of bounds indices from numerical errors
        # idx = idx[0:n_particles-1]
        
        idx = stratified_resampling(w)
        idx.astype(int)
        idx[n_particles - 1] = n_particles - 1

        
        #x_mean_tp1 = jax.vmap(functools.partial(f, F=U[t], dt=dt))(x=X_particles, F_sd=F_particles, p=P_particles) 
        x_mean_tp1 = jax.vmap(functools.partial(f_x, F=U[t], dt=dt))(x=X_particles[...], F_sd=F_particles[...], p=P_particles[...]) 
        x_ref_tp1 = X_prop_im1[:,t+1]

        if ancestor_sampling:
            # (6) sample ancestor of the n_particles-th particle
            logweights = jax.vmap(functools.partial(logweighting, y=x_ref_tp1, cov=X_cov))(x_mean_tp1[idx])
            const = max(logweights)  # Subtract the maximum value for numerical stability
            w_as = np.exp(logweights - const)
            w_as = np.asarray(w_as).astype('float64')
            # w_as = np.squeeze(w_as / sum(w_as))  # Save the normalized weights
            # w_as[n_particles-1] = 0
            # test = w_as/w_as.sum()

            # Fix ancestor sampling! This does not work yet!
            prob = np.multiply(w_as, w)

            # prob = w * jax.vmap(functools.partial(squared_error, y=x_ref_tp1, cov=X_cov))(x_mean_tp1)
            prob = np.asarray(prob).astype('float64')
            prob = np.squeeze(prob / sum(prob))  # Save the normalized weights

            if np.any(np.isnan(prob)):
                plt.subplot(2,2,1)
                for k in range(n_particles):
                    plt.plot(X_store[:,k,0], alpha=w[k], color='b')
                    plt.scatter(t, x_mean_tp1[idx[k],0], marker='x', alpha=w_as[k], color='g')
                plt.plot(X_true[:,0].T,  label='true', markersize=3, color='r')
                plt.plot(X_prop_im1[0,:].T,  label='reference', markersize=3, color='k')
                plt.ylabel('state 0')

                plt.subplot(2,2,3)
                for k in range(n_particles):
                    plt.plot(X_store[:,k,1], alpha=w[k], color='b')
                    plt.scatter(t, x_mean_tp1[idx[k],1], marker='x', alpha=w_as[k], color='g')
                plt.plot(X_true[:,1].T,  label='true', markersize=3, color='r')
                plt.plot(X_prop_im1[1,:].T,  label='reference', markersize=3, color='k')
                plt.ylabel('state 1')
                plt.xlabel('time steps')
                plt.legend()

                plt.subplot(2,2,2)
                for k in range(n_particles):
                    plt.plot(P_store[:,k], alpha=w[k], color='b')
                plt.plot(P_true,  label='true', markersize=3, color='r')
                plt.ylabel('m')

                plt.subplot(2,2,4)
                for k in range(n_particles):
                    plt.plot(F_store[:,k], alpha=w[k], color='b')
                plt.plot(F_true,  label='true', markersize=3, color='r')
                plt.ylabel('F')
                plt.xlabel('time steps')

                # plt.show()
                print(f"Particle degeneration at ancestor sampling in time step {t}.")
                weight_depletion = True
                break
            idx[n_particles-1] = np.random.choice(np.arange(0,n_particles), p=prob)

        
        # (7) sample x for 1,...,n_particles-1
        #X_particles = jax.vmap(functools.partial(f, F_particles=U[t], dt=dt))(x=X_particles[idx,:], F_sd=F_particles[idx,:], p=P_particles[idx,:]) + w_X((n_particles,2))
        X_particles = x_mean_tp1[idx] + w_X((n_particles,))
        
        # (8) sample x for n_particles
        X_particles = X_particles.at[n_particles-1,:].set(x_ref_tp1)

        # (9) sample F_particles
        phi_x = jax.vmap(basis_func)(X_particles)
        F_particles = np.squeeze(jax.vmap(lambda x: jnp.matmul(F_coeff,x))(phi_x)) #+ w_F((n_particles,))

        # (9) sample P_particles
        P_particles = np.ones_like(F_particles)*np.squeeze(P_coeff) #+ w_P((n_particles,))

        # (10) store trajectories for X_particles
        X_store[0:t+2,...] = np.concatenate((X_store[0:t+1,idx,...], X_particles[None,...]),axis=0)

        # (11) store trajectories for F_particles
        F_store[0:t+2,...] = np.concatenate((F_store[0:t+1,idx,...], F_particles[None,...]),axis=0)

        # (11) store trajectories for P_particles
        P_store[0:t+2,...] = np.concatenate((P_store[0:t+1,idx,...], P_particles[None,...]),axis=0)



        # (4) calculate first stage weights
        y_pred = jax.vmap(f_y)(X_particles)
        # w = jax.vmap(functools.partial(squared_error, y=Y[t], cov=R))(y_pred)
        # w = np.asarray(w).astype('float64')
        # w = w/np.sum(w)

        logweights = jax.vmap(functools.partial(logweighting, y=Y[t+1], cov=R))(y_pred)
        # logweights = -(1 / (2 * R)) * (Y[t] - y_pred) ** 2
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w = np.exp(logweights - max_weight)
        w = np.asarray(w).astype('float64')
        w = np.squeeze(w / sum(w))  # Save the normalized weights
        # accumulate the log-likelihood
        # log_likelihood += max_weight + np.log(sum(w)) - np.log(n_particles)

        if np.any(np.isnan(w)):
            plt.subplot(2,2,1)
            #for k in range(n_particles):
                #plt.plot(X_store[:,k,0], alpha=w[k], color='b')
                #plt.scatter(t, x_mean_tp1[idx[k],0], marker='x', alpha=w_as[k], color='g')
            plt.plot(X_true[:,0].T,  label='true', markersize=3, color='r')
            plt.plot(X_prop_im1[0,:].T,  label='reference', markersize=3, color='k')
            plt.ylabel('state 0')

            plt.subplot(2,2,3)
            #for k in range(n_particles):
                #plt.plot(X_store[:,k,1], alpha=w[k], color='b')
                #plt.scatter(t, x_mean_tp1[idx[k],1], marker='x', alpha=w_as[k], color='g')
            plt.plot(X_true[:,1].T,  label='true', markersize=3, color='r')
            plt.plot(X_prop_im1[1,:].T,  label='reference', markersize=3, color='k')
            plt.ylabel('state 1')
            plt.xlabel('time steps')
            plt.legend()

            plt.subplot(2,2,2)
            #for k in range(n_particles):
                #plt.plot(P_store[:,k], alpha=w[k], color='b')
            plt.plot(P_true,  label='true', markersize=3, color='r')
            plt.ylabel('m')

            plt.subplot(2,2,4)
            #for k in range(n_particles):
                #plt.plot(F_store[:,k], alpha=w[k], color='b')
            plt.plot(F_true,  label='true', markersize=3, color='r')
            plt.ylabel('F')
            plt.xlabel('time steps')

            # plt.show()
            print(f"Particle degeneration at weights in time step {t}.")
            weight_depletion = True
            break

    # plt.subplot(2,1,1)
    # #plt.plot(X_store[:,:,0])
    # plt.plot(X_prop_im1[0,:].T)
    # plt.show()




    # (13) draw a sample trajectory as output
    w = np.asarray(w).astype('float64')
    w /= w.sum()
    sample = np.random.choice(np.arange(0,n_particles), p=w)
    X_prop_i = np.squeeze(X_store[:,sample,:]).T
    P_prop_i = np.squeeze(P_store[:,sample]).T
    F_prop_i = np.squeeze(F_store[:,sample]).T

    
    plt.subplot(2,2,1)
    for k in range(n_particles):
        plt.plot(X_store[:,k,0], alpha=w[k], color='b')
    plt.plot(X_true[:,0].T,  label='true', markersize=3, color='r')
    plt.plot(X_prop_im1[0,:].T,  label='reference', markersize=3, color='k')
    plt.plot(X_prop_i[0,:],  label='chosen', markersize=3, color='g')
    plt.ylabel('state 0')

    plt.subplot(2,2,3)
    for k in range(n_particles):
        plt.plot(X_store[:,k,1], alpha=w[k], color='b')
    plt.plot(X_true[:,1].T,  label='true', markersize=3, color='r')
    plt.plot(X_prop_im1[1,:].T,  label='reference', markersize=3, color='k')
    plt.plot(X_prop_i[1,:], label='chosen', markersize=3, color='g')
    plt.ylabel('state 1')
    plt.xlabel('time steps')

    plt.subplot(2,2,2)
    for k in range(n_particles):
        plt.plot(P_store[:,k], alpha=w[k], color='b')
    plt.plot(P_true,  label='true', markersize=3, color='r')
    plt.plot(P_prop_i,  label='chosen', markersize=3, color='g')
    plt.ylabel('m')

    plt.subplot(2,2,4)
    for k in range(n_particles):
        plt.plot(F_store[:,k], alpha=w[k], color='b')
    plt.plot(F_true,  label='true', markersize=3, color='r')
    plt.plot(F_prop_i,  label='chosen', markersize=3, color='g')
    plt.ylabel('F')
    plt.xlabel('time steps')
    
    plt.legend()
    plt.gcf().set_size_inches(10, 5)
    plt.savefig(f"plots/PGAS_iteration_{i}", dpi=400)
    plt.clf()

    # print(f"CSMC successfull in iteration {i}.")
    return X_prop_i, P_prop_i, F_prop_i, weight_depletion





# function for parameter update
def parameter_update(X_prop_i, P_prop_i, F_prop_i,
                    X_prior, P_prior, F_prior,
                    f_x, basis_func, U,
                    dt, seed):
    P_bounds = [0,5]

    ### ---------------------------------------------------------
    ### xi 1 (P)
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    P_out = P_prop_i[None,...]
    P_stats_new = list(prior_niw_calcStatistics(*P_prior, P_out))

    # use suff. statistics to calculate posterior NIW parameters
    P_mean, P_normal_scale, P_iw_scale, P_df = prior_niw_2naturalPara_inv(*P_stats_new)

    # covariance and mean update | state and xi 1 trajectories
    # sample from NIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    # P_cov = invwishart.rvs(
    #             df=int(P_df),
    #             scale=P_iw_scale,
    #             random_state=seed)
    P_cov = np.ones((1,1))*1

    P_cov_col = np.linalg.cholesky(P_cov/P_normal_scale)
    while True:
        n_samples = np.random.normal(size=P_mean.shape)
        P_coeff = P_mean + np.matmul(P_cov_col,n_samples)
        if P_coeff > P_bounds[0] and P_coeff < P_bounds[1]:
            break

    #P_coeff = np.random.multivariate_normal(P_mean,P_cov/P_normal_scale), P_mean.shape

    P_niw_sample = [P_coeff, P_cov]



    ### ---------------------------------------------------------
    ### xi 2 (F)
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    F_in = jax.vmap(basis_func)(X_prop_i.T).T
    F_out = F_prop_i[None,...]
    F_stats_new = list(prior_mniw_calcStatistics(*F_prior, F_in, F_out))
    F_stats_new = [F_stats_new[1],F_stats_new[2],F_stats_new[0],F_stats_new[3]]

    # use suff. statistics to calculate posterior MNIW parameters
    F_mean, F_col_cov, F_iw_scale, F_df = prior_mniw_2naturalPara_inv(*F_stats_new)

    # covariance and mean update | state and xi 2 trajectories
    # sample from MNIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    F_row_cov = invwishart.rvs(
                df=int(F_df),
                scale=F_iw_scale,
                random_state=seed)
    
    
    F_row_cov_col = np.sqrt(F_row_cov)
    F_col_cov_col = np.linalg.cholesky(F_col_cov)
    n_samples = np.random.normal(size=F_mean.shape)
    F_coeff = F_mean + F_row_cov_col*np.matmul(n_samples,F_col_cov_col)
    # F_cov = np.kron(F_col_cov, F_row_cov)
    # F_coeff = np.reshape(np.random.multivariate_normal(F_mean[:],F_cov), F_mean.shape)

    F_mniw_sample = [F_coeff, F_row_cov]



    ### ---------------------------------------------------------
    ### states (X), overall covariance update | trajectories of states and xis 
    ### ---------------------------------------------------------
    # Calculate sufficient statistics with new proposal
    X_in = jax.vmap(functools.partial(f_x, dt=dt))(X_prop_i[:,:-1].T, U[:-1], F_prop_i[:-1], p=P_prop_i[:-1]).T
    X_out = X_prop_i[:,1:]
    X_stats_new = prior_iw_calcStatistics(*X_prior, X_in, X_out)

    # use suff. statistics to calculate posterior IW parameters
    X_iw_scale, X_df = prior_iw_2naturalPara_inv(*X_stats_new)

    # covariance and mean update | state and xi 1 trajectories
    # sample from MNIW proposal using posterior parameters
    # Note, in the offline algorithm, it is not required to sample from the predictive distribution but from the posterior in order to obtain parameter samples!
    X_cov = invwishart.rvs(
                df=int(X_df),
                scale=X_iw_scale,
                random_state=seed)
    
    X_iw_sample = [X_cov]


    return X_iw_sample, P_niw_sample, F_mniw_sample













if __name__ == '__main__':
    from matplotlib import pyplot as plt

    ################################################################################
    # General settings

    n_particles = 50    # no of particles
    t_end = 50.0
    dt = 0.04
    time = np.arange(0.0,t_end,dt)
    steps = len(time)
    seed = 123 #np.random.randint(100, 1000000)
    rng = np.random.default_rng()
    print(f"Seed is: {seed}")
    np.random.seed(seed)



    ################################################################################
    # Generating offline training data

    # initial system state
    x0 = np.array([0.0, 0.0])
    P0 = np.diag([5e-8, 5e-9])


    # noise
    R = np.array([[1e-3]])
    Q = np.diag([5e-8, 5e-9])
    w = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), Q, n)
    e = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), R, n)


    # time series for plot
    X = np.zeros((steps,2)) # sim
    X[0,...] = x0
    Y = np.zeros((steps,))
    F_sd = np.zeros((steps,))
    M = np.ones((steps,))*m

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
    ### Initialize prior values for parameters

    # -----------------
    # X: parameters of the iw prior
    X_iw_scale = 1e1*np.eye(2) # np.diag([1e1,1e2]) # defining distribution parameters
    X_df = 2

    X_cov = invwishart.rvs(df=X_df, scale=X_iw_scale, random_state=seed) # sampling 
    X_iw_sample = [X_cov]

    X_prior = [X_iw_scale, X_df] # storing natural parameters



    # -----------------
    # P: parameters of the niw prior
    P_mean = 2*np.eye(1) # defining distribution parameters
    P_normal_scale = 1e3*np.eye(1)
    P_iw_scale = 1e0*np.eye(1)
    P_df = 1

    P_cov = np.array([[invwishart.rvs(df=P_df, scale=P_iw_scale, random_state=seed)]]) # sampling 
    P_cov_col = np.linalg.cholesky(P_cov/P_normal_scale)
    n_samples = np.random.normal(size=P_mean.shape)
    P_coeff = P_mean + np.matmul(P_cov_col,n_samples)
    P_niw_sample = [P_coeff, P_cov]

    eta_0, eta_1, eta_2, eta_3  = prior_niw_2naturalPara(P_mean, P_normal_scale, P_iw_scale, P_df) # storing natural parameters
    P_prior = [eta_0, eta_1, eta_2, eta_3]



    # -----------------
    # F: parameters of the mniw prior
    F_mean = np.zeros((1, N_ip)) # defining distribution parameters
    # F_col_cov = np.eye(N_ip)*1e0 ### This needs to be adjusted!
    F_col_cov = np.diag(sd)
    F_iw_scale = 1e2*np.eye(1)
    F_df = 1

    F_row_cov = np.array([[invwishart.rvs(df=F_df, scale=F_iw_scale, random_state=seed)]]) # sampling 
    F_row_cov_col = np.sqrt(F_row_cov)
    F_col_cov_col = np.linalg.cholesky(F_col_cov)
    n_samples = np.random.normal(size=F_mean.shape)
    F_coeff = F_mean + F_row_cov_col*np.matmul(n_samples,F_col_cov_col)
    F_mniw_sample = [F_coeff, F_row_cov]

    eta_1, eta_2, eta_0, eta_3 = prior_mniw_2naturalPara(F_mean, F_col_cov, F_iw_scale, F_df) # storing natural parameters
    F_prior = [eta_0, eta_1, eta_2, eta_3]


    # -----------------------------------------------------------------------------------
    ### Initialize first reference trajectory for X
    X_prop_im1 = np.zeros((2,steps)) 
    # N_init = int(2e4)
    # X_prop_im1[...,0] = x0
    # for t in range(1,steps):
    #     F = np.matmul(F_coeff,basis_fcn(X_prop_im1[:,t-1]))
    #     X_prop_im1[:,t:t+1] = f_x(X_prop_im1[:,t-1:t], U[t-1], F, dt=dt, p=np.squeeze(P_coeff))
        
    # X_prop_im1 = X.T + 1e-1*np.random.normal(size=X.T.shape)

    # ancestor_sampling = True
    # N_init = 1e2
    # resampling_method = 'multi'
    # X_prop_im1 = np.zeros((2,steps)) 

    # csmc_instance = CSMC(f_x, f_y, basis_fcn, dt, R,
    #                                         P_niw_sample, F_mniw_sample, X_iw_sample)
    # csmc_instance.sample_states(Y, U, x0, X_prop_im1, N_init, ancestor_sampling,
    #                                         resampling_method, 1)
    

    ### Step 0: Get new state trajectory proposal from PGAS Markov kernel with bootstrap PF | Given the parameters
    # X_prop_im1 = np.zeros((2,steps)) 
    # N_init = int(1e2)
    # for i in range(10):
    #     X_prop_im1, _, __ = conditional_SMC_with_ancestor_sampling(Y, U,                     # observations and inputs
    #                                                         X_iw_sample, P_niw_sample, F_mniw_sample,   # parameters
    #                                                         x0, X_prop_im1,                                 # reference trajectory
    #                                                         f_x, f_y, basis_fcn, R,                             # model structures (including known covariance R)
    #                                                         N_init, dt)     

    ### Step 1: Get new state trajectory proposal from PGAS Markov kernel with bootstrap PF | Given the parameters
    P_niw_sample = [np.eye(1)*m, np.eye(1)]
    X_prop_im1, _, __, weight_depletion = conditional_SMC_with_ancestor_sampling(Y, U,                     # observations and inputs
                                                        X_iw_sample, P_niw_sample, F_mniw_sample,   # parameters
                                                        x0, X_prop_im1,                                 # reference trajectory
                                                        f_x, f_y, basis_fcn, R,                             # model structures (including known covariance R)
                                                        100, dt, X, M, F_sd, 0, ancestor_sampling = False)                                      # further settings

    # -----------------------------------------------------------------------------------
    ### identification loop
    for i in tqdm(range(1,steps), desc="Running identification"):
        weight_depletion = True
        P_niw_sample = [np.eye(1)*m, np.eye(1)]
        while weight_depletion:
            ### Step 1: Get new state trajectory proposal from PGAS Markov kernel with bootstrap PF | Given the parameters
            X_prop_i, P_prop_i, F_prop_i, weight_depletion = conditional_SMC_with_ancestor_sampling(Y, U,                     # observations and inputs
                                                                X_iw_sample, P_niw_sample, F_mniw_sample,   # parameters
                                                                x0, X_prop_im1,                                 # reference trajectory
                                                                f_x, f_y, basis_fcn, R,                             # model structures (including known covariance R)
                                                                n_particles, dt, X, M, F_sd, i, ancestor_sampling = True)                                      # further settings
        print(f"CSMC successfull in iteration {i}.")
        
        ### Step 2: Compute parameter posterior | Given the state trajectory
        X_iw_sample, P_niw_sample, F_mniw_sample = parameter_update(X_prop_i, P_prop_i, F_prop_i,       # samples for the latent variable sequences
                                                                X_prior, P_prior, F_prior,              # initial values for parameter priors
                                                                f_x, basis_fcn, U,                              # model structures (including known covariance R)
                                                                dt, seed)                               # further settings
        
        X_prop_im1 = X_prop_i

    # ToDo: 
    # 1. Animation plots
    # 2. Finding initial reference trajectory and/or initialization? --> Running normal SMC at start?


    ################################################################################
    # Plots

    # fig = generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, basis_fcn, W, CW, time, 200., 30., 30)

    # print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-Sigma_F)))

    # fig.show()