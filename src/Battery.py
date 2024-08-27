import jax
import jax.numpy as jnp
import numpy as np
import functools
import pandas as pd
from tqdm import tqdm



from src.BayesianInferrence import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_calcStatistics
from src.BayesianInferrence import prior_mniw_Predictive
from src.Filtering import systematic_SISR, log_likelihood_Normal, log_likelihood_Multivariate_t



#### This section defines the state space model

# parameters
V_0 = 2.4125 # interpolated from product specification V_nom at 1.0C and V_cutoff at 0.2C
R_0 = 23e-3



# state dynamics
def dx(x, I, R_1, C_1):
        
        dV = I / C_1 - x / R_1 / C_1
    
        return dV



# time discrete state space model with Runge-Kutta-4
@jax.jit
def f_x(x, I, R_1, C_1, dt):
    
        C_1 = (offset_C1 + C_1) * scale_C1
        R_1 = (offset_R1 + R_1) * scale_R1
        
        k1 = dx(x, I, R_1, C_1)
        k2 = dx(x+dt*k1/2, I, R_1, C_1)
        k3 = dx(x+dt*k2/2, I, R_1, C_1)
        k4 = dx(x+dt*k3, I, R_1, C_1)
        
        return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)



# measurement model
@jax.jit
def f_y(x, I, R_0=R_0, V_0=V_0):
    return V_0 + x + R_0*I



@jax.jit
def log_likelihood(obs_x, obs_C1R1, x_mean, x_1, C1R1_1, Mean_C1R1, Col_Cov_C1R1, Row_Scale_C1R1, df_C1R1):
    
    # log likelihood of state x
    l_x = log_likelihood_Normal(obs_x, x_mean, Q)
    
    # log likelihood of force F_sd from conditional predictive PDF
    basis = basis_fcn(obs_x)
    c_mean, c_col_scale, c_row_scale, c_df = prior_mniw_Predictive(
        mean=Mean_C1R1,
        col_cov=Col_Cov_C1R1,
        row_scale=Row_Scale_C1R1,
        df=df_C1R1,
        y1=C1R1_1,
        basis=basis
        )
    
    l_F = log_likelihood_Multivariate_t(
        observed=obs_C1R1, 
        mean=c_mean, 
        scale=c_col_scale*c_row_scale, 
        df=c_df
        )
    
    return l_x + l_F



#### This section loads the measurment data

### Load data
time_gap_thresh = pd.Timedelta("1h")
data = pd.read_csv("./src/Measurements/Everlast_35E_003.csv", sep=",")
data["Time"] = pd.to_datetime(data["Time"])
data = data.set_index("Time")

# segment data
data["Block"] = (data.index.to_series().diff() > time_gap_thresh).cumsum()
blocks = [group for _, group in data.groupby("Block")]
data = blocks[3]
del blocks

# resampling to uniform time steps
dt = pd.Timedelta("1s")
data = data.resample("5ms", origin="start").interpolate().resample("1s", origin="start").asfreq()

# data = data.iloc[:2000]
steps = data.shape[0]



#### This section defines relevant parameters for the simulation

# set a seed for reproducability
rng = np.random.default_rng(16723573)

# sim para
N_particles = 100
N_PGAS_iter = 5
forget_factor = 1 - 1/1e3
burnin_iter = 100
dt = dt.seconds


# parameter limits
l_C1, u_C1 = 2, 20
l_R1, u_R1 = 5, 10

# limits for clipping
offset_C1 = (u_C1 - l_C1)/2
offset_R1 = (u_R1 - l_R1)/2
cl_C1, cu_C1 = l_C1 - offset_C1, u_C1 - offset_C1
cl_R1, cu_R1 = l_R1 - offset_R1, u_R1 - offset_R1
scale_C1 = 1e2
scale_R1 = 1e3


# initial system state
x0 = np.array([data["Voltage"].iloc[0]-V_0])
P0 = np.diag([5e-3])

# process and measurement noise
R = np.diag([0.001])
Q = np.diag([5e-6])
w = lambda n=1: rng.multivariate_normal(np.zeros((Q.shape[0],)), Q, n)
e = lambda n=1: rng.multivariate_normal(np.zeros((R.shape[0],)), R, n)

# input
ctrl_input = data["Current"].to_numpy()

# measurments
Y = data[["Voltage"]].to_numpy()
time = data.index
steps = Y.shape[0]


#### This section defines the basis function expansion

N_basis_fcn = 15
basis_fcn, sd = generate_Hilbert_BasisFunction(N_basis_fcn, jnp.array([-0.5, 2.4]), 2.4/N_basis_fcn, 5)

GP_prior = list(prior_mniw_2naturalPara(
    np.zeros((2, N_basis_fcn)),
    np.diag(sd),
    np.diag([1, 1]),
    0
))



#### This section defines a function to run the online version of the algorithm

def Battery_APF(Y=Y):
    
    print("\n=== Online Algorithm ===")
    
    # time series for plot
    Sigma_X = np.zeros((steps,N_particles))
    Sigma_Y = np.zeros((steps,N_particles))
    Sigma_C1R1 = np.zeros((steps,N_particles,2))
    
    # variable for sufficient statistics
    GP_stats = [
        np.zeros((N_particles, N_basis_fcn, 2)),
        np.zeros((N_particles, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_particles, 2, 2)),
        np.zeros((N_particles,))
    ]
    GP_stats_logging = [
        np.zeros((steps, N_basis_fcn, 2)),
        np.zeros((steps, N_basis_fcn, N_basis_fcn)),
        np.zeros((steps, 2, 2)),
        np.zeros((steps,))
    ]
    
    # initial values for states
    Sigma_X[0,...] = rng.multivariate_normal(x0, P0, (N_particles,)).flatten()
    Sigma_C1R1[0,:,0] = rng.uniform(cl_C1, cu_C1, (N_particles,))
    Sigma_C1R1[0,:,1] = rng.uniform(cl_R1, cu_R1, (N_particles,))
    weights = np.ones((steps,N_particles))/N_particles

    # update GP
    phi_0 = jax.vmap(
        functools.partial(basis_fcn)
        )(Sigma_X[0])
    GP_stats = list(jax.vmap(prior_mniw_calcStatistics)(
        Sigma_C1R1[0,...],
        phi_0
    ))
    
    # logging
    GP_stats_logging[0][0] = np.einsum('n...,n->...', GP_stats[0], weights[0])
    GP_stats_logging[1][0] = np.einsum('n...,n->...', GP_stats[1], weights[0])
    GP_stats_logging[2][0] = np.einsum('n...,n->...', GP_stats[2], weights[0])
    GP_stats_logging[3][0] = np.einsum('n...,n->...', GP_stats[3], weights[0])
    
    
    # simulation loop
    for i in tqdm(range(1, steps), desc="Running Online Algorithm"):
        
        ### Step 1: Propagate GP parameters in time
        
        # calculate effective forgetting factor
        f_forget = np.minimum((forget_factor-0.2)*(1-i/burnin_iter) + forget_factor*i/burnin_iter, forget_factor)
        
        # apply forgetting operator to statistics for t-1 -> t
        GP_stats[0] *= f_forget
        GP_stats[1] *= f_forget
        GP_stats[2] *= f_forget
        GP_stats[3] *= f_forget
            
        # calculate parameters of GP from prior and sufficient statistics
        GP_para = list(jax.vmap(prior_mniw_2naturalPara_inv)(
            GP_prior[0] + GP_stats[0],
            GP_prior[1] + GP_stats[1],
            GP_prior[2] + GP_stats[2],
            GP_prior[3] + GP_stats[3]
        ))
        
        
        
        ### Step 2: According to the algorithm of the auxiliary PF, resample 
        # particles according to the first stage weights
        
        # create auxiliary variable for state x
        x_aux = jax.vmap(
            functools.partial(f_x, I=ctrl_input[i-1], dt=dt))(
                x=Sigma_X[i-1],
                C_1=Sigma_C1R1[i-1,...,0],
                R_1=Sigma_C1R1[i-1,...,1]
            )
        
        # create auxiliary variable for C1, R1 and R0
        basis = jax.vmap(functools.partial(basis_fcn))(x_aux)
        C1R1_aux = jax.vmap(
            functools.partial(prior_mniw_Predictive)
            )(
                mean=GP_para[0],
                col_cov=GP_para[1],
                row_scale=GP_para[2],
                df=GP_para[3],
                basis=basis
        )[0]
        
        # calculate first stage weights
        y_aux = jax.vmap(functools.partial(f_y, I=ctrl_input[i]))(x=x_aux)
        l_fy = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(y_aux)
        l_C0_clip = np.all(np.vstack([cl_C1 <= C1R1_aux[:,0], C1R1_aux[:,0] <= cu_C1]), axis=0)
        l_R0_clip = np.all(np.vstack([cl_R1 <= C1R1_aux[:,1], C1R1_aux[:,1] <= cu_R1]), axis=0)
        weights_aux = weights[i-1] * np.exp(l_fy) * l_C0_clip * l_R0_clip
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
        w_x = w((N_particles,)).flatten()
        Sigma_X[i] = jax.vmap(
            functools.partial(f_x, I=ctrl_input[i-1], dt=dt))(
                x=Sigma_X[i-1,idx],
                C_1=Sigma_C1R1[i-1,idx,0],
                R_1=Sigma_C1R1[i-1,idx,1]
            ) + w_x
        
        ## sample proposal for alpha and beta at time t
        # evaluate basis functions
        basis = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i])
        
        # calculate conditional predictive distribution for C1 and R1
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
        c_col_scale_chol = np.linalg.cholesky(c_col_scale)
        c_row_scale_chol = np.linalg.cholesky(c_row_scale)
        t_samples = rng.standard_t(df=df, size=(1,2,N_particles)).T
        Sigma_C1R1[i] = c_mean + np.squeeze(np.einsum(
                '...ij,...jk,...kf->...if', 
                c_row_scale_chol, 
                t_samples, 
                c_col_scale_chol
            ))
            
            
        
        # Update the sufficient statistics of GP with new proposal
        T_0, T_1, T_2, T_3 = list(jax.vmap(prior_mniw_calcStatistics)(
            Sigma_C1R1[i],
            basis
        ))
        GP_stats[0] = GP_stats[0][idx] + T_0
        GP_stats[1] = GP_stats[1][idx] + T_1
        GP_stats[2] = GP_stats[2][idx] + T_2
        GP_stats[3] = GP_stats[3][idx] + T_3
        
        # calculate new weights
        Sigma_Y[i] = jax.vmap(functools.partial(f_y, I=ctrl_input[i]))(x=Sigma_X[i])
        q_fy = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(Sigma_Y[i])
        q_C0_clip = np.all(np.vstack([cl_C1 <= Sigma_C1R1[i,:,0], Sigma_C1R1[i,:,0] <= cu_C1]), axis=0)
        q_R0_clip = np.all(np.vstack([cl_R1 <= Sigma_C1R1[i,:,1], Sigma_C1R1[i,:,1] <= cu_R1]), axis=0)
        weights[i] = np.exp(q_fy) * q_C0_clip * q_R0_clip / weights_aux[idx]
        weights[i] = weights[i]/np.sum(weights[i])
        
        
        
        # logging
        GP_stats_logging[0][i] = np.einsum('n...,n->...', GP_stats[0], weights[i])
        GP_stats_logging[1][i] = np.einsum('n...,n->...', GP_stats[1], weights[i])
        GP_stats_logging[2][i] = np.einsum('n...,n->...', GP_stats[2], weights[i])
        GP_stats_logging[3][i] = np.einsum('n...,n->...', GP_stats[3], weights[i])
        
        #abort
        if np.any(np.isnan(weights[i])):
            print("Particle degeneration at new weights")
            break
    
    return Sigma_X, Sigma_C1R1, Sigma_Y, weights, GP_stats_logging



#### This section defines a function to run the offline version of the algorithm

def Battery_PGAS():
    
    print("\n=== Offline Algorithm ===")
    
    Sigma_X = np.zeros((steps,N_PGAS_iter))
    Sigma_C1R1 = np.zeros((steps,N_PGAS_iter,2))
    Sigma_Y = np.zeros((steps,N_PGAS_iter,))
    weights = np.ones((steps,N_PGAS_iter))/N_PGAS_iter

    # variable for sufficient statistics
    GP_stats = [
        np.zeros((N_PGAS_iter, N_basis_fcn, 2)),
        np.zeros((N_PGAS_iter, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_PGAS_iter, 2, 2)),
        np.zeros((N_PGAS_iter,))
    ]

    # set initial reference using APF
    print(f"Setting initial reference trajectory")
    GP_para = prior_mniw_2naturalPara_inv(
        GP_prior[0],
        GP_prior[1],
        GP_prior[2],
        GP_prior[3]
    )
    Sigma_X[:,0], Sigma_C1R1[:,0], Sigma_Y[:,0] = Battery_CPFAS_Kernel(
            x_ref=None,
            C1R1_ref=None,
            Mean_C1R1=GP_para[0], 
            Col_Cov_C1R1=GP_para[1], 
            Row_Scale_C1R1=GP_para[2], 
            df_C1R1=GP_para[3])
        
        
    # make proposal for distribution of F_sd using new proposals of trajectories
    phi = jax.vmap(basis_fcn)(Sigma_X[:,0])
    T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_C1R1[:,0],
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
        Sigma_X[:,k], Sigma_C1R1[:,k], Sigma_Y[:,k] = Battery_CPFAS_Kernel(
            x_ref=Sigma_X[:,k-1],
            C1R1_ref=Sigma_C1R1[:,k-1],
            Mean_C1R1=GP_para[0], 
            Col_Cov_C1R1=GP_para[1], 
            Row_Scale_C1R1=GP_para[2], 
            df_C1R1=GP_para[3])
        
        
        # make proposal for distribution of F_sd using new proposals of trajectories
        phi = jax.vmap(basis_fcn)(Sigma_X[:,k])
        T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
                Sigma_C1R1[:,k],
                phi
            )
        GP_stats[0][k] = np.sum(T_0, axis=0)
        GP_stats[1][k] = np.sum(T_1, axis=0)
        GP_stats[2][k] = np.sum(T_2, axis=0)
        GP_stats[3][k] = np.sum(T_3, axis=0)
        
    
    return Sigma_X, Sigma_C1R1, Sigma_Y, weights, GP_stats



def Battery_CPFAS_Kernel(x_ref, C1R1_ref, Mean_C1R1, Col_Cov_C1R1, Row_Scale_C1R1, df_C1R1, Y=Y):
    
    # time series for plot
    Sigma_X = np.zeros((steps,N_particles))
    Sigma_C1R1 = np.zeros((steps,N_particles,2))
    Sigma_Y = np.zeros((steps,N_particles))
    ancestor_idx = np.zeros((steps-1,N_particles))
    
    # initial values for states
    Sigma_X[0,...] = rng.multivariate_normal(x0, P0, (N_particles,)).flatten()
    Sigma_C1R1[0,:,0] = rng.uniform(cl_C1, cu_C1, (N_particles,))
    Sigma_C1R1[0,:,1] = rng.uniform(cl_R1, cu_R1, (N_particles,))
    weights = np.ones((steps,N_particles))/N_particles
    
    if x_ref is not None:
       Sigma_X[0,-1] = x_ref[0]
       Sigma_C1R1[0,-1] = C1R1_ref[0]
    
    
    # simulation loop
    for i in tqdm(range(1, steps), desc="    Running CPF Kernel"):
        
        
        
        ### Step 1: According to the algorithm of the auxiliary PF, resample 
        # particles according to the first stage weights
        
        # create auxiliary variable for state x
        x_aux = jax.vmap(
            functools.partial(f_x, I=ctrl_input[i-1], dt=dt))(
                x=Sigma_X[i-1],
                C_1=Sigma_C1R1[i-1,...,0],
                R_1=Sigma_C1R1[i-1,...,1]
            )
        
        # create auxiliary variable for C1, R1 and R0
        basis = jax.vmap(functools.partial(basis_fcn))(x_aux)
        C1R1_aux = jax.vmap(
            functools.partial(
                prior_mniw_Predictive, 
                mean=Mean_C1R1,
                col_cov=Col_Cov_C1R1,
                row_scale=Row_Scale_C1R1,
                df=df_C1R1)
            )(
                basis=basis
        )[0]
        
        # calculate first stage weights
        y_aux = jax.vmap(functools.partial(f_y, I=ctrl_input[i]))(x=x_aux)
        l_fy = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(y_aux)
        l_C0_clip = np.all(np.vstack([cl_C1 <= C1R1_aux[:,0], C1R1_aux[:,0] <= cu_C1]), axis=0)
        l_R0_clip = np.all(np.vstack([cl_R1 <= C1R1_aux[:,1], C1R1_aux[:,1] <= cu_R1]), axis=0)
        weights_aux = weights[i-1] * l_fy * l_C0_clip * l_R0_clip
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
        w_x = w((N_particles,)).flatten()
        Sigma_X_mean = jax.vmap(
            functools.partial(f_x, I=ctrl_input[i-1], dt=dt))(
                x=Sigma_X[i-1,idx],
                C_1=Sigma_C1R1[i-1,idx,0],
                R_1=Sigma_C1R1[i-1,idx,1]
            )
        Sigma_X[i] = Sigma_X_mean + w_x
        
        # set reference trajectory for state x
        if x_ref is not None:
            Sigma_X[i,-1] = x_ref[i]
        
        ## sample proposal for alpha and beta at time t
        # evaluate basis functions
        basis = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i])
        
        # calculate conditional predictive distribution for C1 and R1
        c_mean, c_col_scale, c_row_scale, df = jax.vmap(
            functools.partial(
                prior_mniw_Predictive, 
                mean=Mean_C1R1,
                col_cov=Col_Cov_C1R1,
                row_scale=Row_Scale_C1R1,
                df=df_C1R1)
            )(
                basis=basis
        )
        
        # generate samples
        c_col_scale_chol = np.linalg.cholesky(c_col_scale)
        c_row_scale_chol = np.linalg.cholesky(c_row_scale)
        t_samples = rng.standard_t(df=df, size=(1,2,N_particles)).T
        Sigma_C1R1[i] = c_mean + np.squeeze(np.einsum(
                '...ij,...jk,...kf->...if', 
                c_row_scale_chol, 
                t_samples, 
                c_col_scale_chol
            ))
        
        # set reference trajectory for C1 and R1
        if x_ref is not None:
            Sigma_C1R1[i,-1] = C1R1_ref[i]
            
        
        
        ### Step 3: Sample a new ancestor for the reference trajectory
        
        if x_ref is not None:
            
            # calculate ancestor weights
            l_x = jax.vmap(
                functools.partial(
                    log_likelihood, 
                    obs_x=Sigma_X[i,-1], 
                    obs_C1R1=Sigma_C1R1[i,-1], 
                    Mean_C1R1=Mean_C1R1, 
                    Col_Cov_C1R1=Col_Cov_C1R1, 
                    Row_Scale_C1R1=Row_Scale_C1R1, 
                    df_C1R1=df_C1R1)
                )(x_mean=Sigma_X_mean, x_1=Sigma_X[i-1], C1R1_1=Sigma_C1R1[i-1])
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
        Sigma_Y[i] = jax.vmap(functools.partial(f_y, I=ctrl_input[i]))(x=Sigma_X[i])
        q_fy = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(Sigma_Y[i])
        q_C0_clip = np.all(np.vstack([cl_C1 <= Sigma_C1R1[i,:,0], Sigma_C1R1[i,:,0] <= cu_C1]), axis=0)
        q_R0_clip = np.all(np.vstack([cl_R1 <= Sigma_C1R1[i,:,1], Sigma_C1R1[i,:,1] <= cu_R1]), axis=0)
        weights[i] = np.exp(q_fy) * q_C0_clip * q_R0_clip / weights_aux[idx]
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
    x_traj = np.zeros((steps))
    x_traj[-1] = Sigma_X[-1, idx_traj]
    C1R1_traj = np.zeros((steps,2))
    C1R1_traj[-1] = Sigma_C1R1[-1, idx_traj]
    y_traj = np.zeros((steps))
    y_traj[-1] = Sigma_Y[-1, idx_traj]
    ancestry = np.zeros((steps,))
    ancestry[-1] = idx_traj
    for i in range(steps-2, -1, -1): # run backward in time
        ancestry[i] = ancestor_idx[i, int(ancestry[i+1])]
        x_traj[i] = Sigma_X[i, int(ancestry[i])]
        C1R1_traj[i] = Sigma_C1R1[i, int(ancestry[i])]
        y_traj[i] = Sigma_Y[i, int(ancestry[i])]
    
    return x_traj, C1R1_traj, y_traj