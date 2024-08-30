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
from src.Filtering import systematic_SISR, log_likelihood_Normal
from src.Filtering import log_likelihood_Multivariate_t, reconstruct_trajectory



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
def log_likelihood(obs_x, obs_C1R1, x_mean, Mean, Col_Cov, Row_Scale, df):
    
    # log likelihood of state x
    l_x = log_likelihood_Normal(obs_x, x_mean, Q)
    
    # log likelihood from predictive PDF
    basis = basis_fcn(obs_x)
    c_mean, c_col_scale, c_row_scale, c_df = prior_mniw_Predictive(
        mean=Mean,
        col_cov=Col_Cov,
        row_scale=Row_Scale,
        df=df,
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
forget_factor = 0.999
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
    
    # time series for plot
    Sigma_X = np.zeros((steps,N_particles))
    Sigma_Y = np.zeros((steps,N_particles))
    Sigma_C1R1 = np.zeros((steps,N_particles,2))
    log_weights = np.zeros((steps,N_particles))
    
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

    # update GP
    basis = jax.vmap(
        functools.partial(basis_fcn)
        )(Sigma_X[0])
    GP_stats = list(jax.vmap(prior_mniw_calcStatistics)(
        Sigma_C1R1[0,...],
        basis
    ))
    
    # logging
    weights = np.exp(log_weights[0] - np.max(log_weights[0]))
    weights /= np.sum(weights)
    GP_stats_logging[0][0] = np.einsum('n...,n->...', GP_stats[0], weights)
    GP_stats_logging[1][0] = np.einsum('n...,n->...', GP_stats[1], weights)
    GP_stats_logging[2][0] = np.einsum('n...,n->...', GP_stats[2], weights)
    GP_stats_logging[3][0] = np.einsum('n...,n->...', GP_stats[3], weights)
    
    
    # simulation loop
    for i in tqdm(range(1, steps), desc="Running APF Algorithm"):
        
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
            functools.partial(f_x, I=ctrl_input[i-1], dt=dt)
            )(
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
        l_fy_aux = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(y_aux)
        
        l_C0_clip_aux = np.all(np.vstack([cl_C1 >= C1R1_aux[:,0], C1R1_aux[:,0] >= cu_C1]), axis=0)
        l_C0_clip_aux = np.array(l_C0_clip_aux, dtype=np.float_) * -500
        l_R0_clip_aux = np.all(np.vstack([cl_R1 >= C1R1_aux[:,1], C1R1_aux[:,1] >= cu_R1]), axis=0)
        l_R0_clip_aux = np.array(l_R0_clip_aux, dtype=np.float_) * -500
        
        log_weights_aux = log_weights[i-1] + l_fy_aux + l_C0_clip_aux + l_R0_clip_aux
        weights_aux = np.exp(log_weights_aux - np.max(log_weights_aux))
        weights_aux = weights_aux/np.sum(weights_aux)
        
        #abort
        if np.any(np.isnan(weights_aux)):
            print("Particle degeneration at auxiliary weights")
            break
        
        # draw new indices
        idx = np.array(systematic_SISR(rng.random(), weights_aux))
        # correct out of bounds indices from numerical errors
        idx[idx >= N_particles] = N_particles - 1 
        
        
        
        ### Step 3: Make a proposal by generating samples from the hirachical 
        # model
        
        # sample from proposal for x at time t
        Sigma_X[i] = jax.vmap(
            functools.partial(f_x, I=ctrl_input[i-1], dt=dt)
            )(
            x=Sigma_X[i-1,idx],
            C_1=Sigma_C1R1[i-1,idx,0],
            R_1=Sigma_C1R1[i-1,idx,1]
        ) + w((N_particles,)).flatten()
        
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
        l_fy = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(Sigma_Y[i])
        
        l_C0_clip = np.all(np.vstack([cl_C1 >= Sigma_C1R1[i,:,0], Sigma_C1R1[i,:,0] >= cu_C1]), axis=0)
        l_C0_clip = np.array(l_C0_clip, dtype=np.float_) * -500
        l_R0_clip = np.all(np.vstack([cl_R1 >= Sigma_C1R1[i,:,1], Sigma_C1R1[i,:,1] >= cu_R1]), axis=0)
        l_R0_clip = np.array(l_R0_clip, dtype=np.float_) * -500
        
        log_weights[i] = l_fy + l_C0_clip + l_R0_clip - l_fy_aux[idx] - l_C0_clip_aux[idx] - l_R0_clip_aux[idx]
        
        
        
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
    
    return Sigma_X, Sigma_C1R1, Sigma_Y, weights, GP_stats_logging



#### This section defines a function to run the offline version of the algorithm

def Battery_PGAS():
    
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
    init_X, init_C1R1, _, init_w, _ = Battery_APF()
    idx = np.searchsorted(np.cumsum(init_w[-1]), rng.random())
    Sigma_X[:,0] = init_X[:,idx]
    Sigma_C1R1[:,0] = init_C1R1[:,idx]
    
        
        
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
            GP_stats_ref=[
                GP_stats[0][k-1],
                GP_stats[1][k-1],
                GP_stats[2][k-1],
                GP_stats[3][k-1]
            ]
        )
        
        
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



def Battery_CPFAS_Kernel(x_ref, C1R1_ref, GP_stats_ref, Y=Y):
    
    # time series for plot
    Sigma_X = np.zeros((steps,N_particles))
    Sigma_C1R1 = np.zeros((steps,N_particles,2))
    Sigma_Y = np.zeros((steps,N_particles))
    ancestor_idx = np.zeros((steps-1,N_particles))
    log_weights = np.zeros((steps,N_particles))
    
    # initial values for states
    Sigma_X[0,...] = rng.multivariate_normal(x0, P0, (N_particles,)).flatten()
    Sigma_C1R1[0,:,0] = rng.uniform(cl_C1, cu_C1, (N_particles,))
    Sigma_C1R1[0,:,1] = rng.uniform(cl_R1, cu_R1, (N_particles,))
    
    Sigma_X[0,-1] = x_ref[0]
    Sigma_C1R1[0,-1] = C1R1_ref[0]
    
    
    
    ## split model into reference and ancestor statistics
    
    # calculate ancestor statistics
    basis = jax.vmap(basis_fcn)(Sigma_X[0,...])
    GP_stats_ancestor = list(jax.vmap(prior_mniw_calcStatistics)(
        Sigma_C1R1[0,...],
        basis
    ))
    
    # update reference statistic
    basis = basis_fcn(x_ref[0])
    T_0, T_1, T_2, T_3 = prior_mniw_calcStatistics(C1R1_ref[0], basis)
    GP_stats_ref[0] -= T_0
    GP_stats_ref[1] -= T_1
    GP_stats_ref[2] -= T_2
    GP_stats_ref[3] -= T_3
    
    
    # simulation loop
    for i in tqdm(range(1, steps), desc="    Running CPF Kernel"):
        
        # calculate models
        Mean, Col_Cov, Row_Scale, df = jax.vmap(prior_mniw_2naturalPara_inv)(
            GP_prior[0] + GP_stats_ref[0] + GP_stats_ancestor[0],
            GP_prior[1] + GP_stats_ref[1] + GP_stats_ancestor[1],
            GP_prior[2] + GP_stats_ref[2] + GP_stats_ancestor[2],
            GP_prior[3] + GP_stats_ref[3] + GP_stats_ancestor[3],
        )
        
        
        
        ### Step 1: According to the algorithm of the auxiliary PF, sample new 
        # ancestor indices according to the first stage weights
        
        # create auxiliary variable for state x
        x_aux = jax.vmap(
            functools.partial(f_x, I=ctrl_input[i-1], dt=dt)
            )(
            x=Sigma_X[i-1],
            C_1=Sigma_C1R1[i-1,...,0],
            R_1=Sigma_C1R1[i-1,...,1]
        )
        
        # create auxiliary variable for C1, R1 and R0
        basis = jax.vmap(functools.partial(basis_fcn))(x_aux)
        C1R1_aux = jax.vmap(
            functools.partial(prior_mniw_Predictive)
            )(
            basis=basis, 
            mean=Mean,
            col_cov=Col_Cov,
            row_scale=Row_Scale,
            df=df
        )[0]
        
        # calculate first stage weights
        y_aux = jax.vmap(functools.partial(f_y, I=ctrl_input[i]))(x=x_aux)
        l_fy_aux = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(y_aux)
        
        l_C0_clip_aux = np.all(np.vstack([cl_C1 >= C1R1_aux[:,0], C1R1_aux[:,0] >= cu_C1]), axis=0)
        l_C0_clip_aux = np.array(l_C0_clip_aux, dtype=np.float_) * -500
        l_R0_clip_aux = np.all(np.vstack([cl_R1 >= C1R1_aux[:,1], C1R1_aux[:,1] >= cu_R1]), axis=0)
        l_R0_clip_aux = np.array(l_R0_clip_aux, dtype=np.float_) * -500
        
        log_weights_aux = log_weights[i-1] + l_fy_aux + l_C0_clip_aux + l_R0_clip_aux
        weights_aux = np.exp(log_weights_aux - np.max(log_weights_aux))
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
        
        
        
        ### Step 2: Sample a new ancestor for the reference trajectory
        
        # calculate ancestor weights
        l_x = jax.vmap(
            functools.partial(
            log_likelihood, 
            obs_x=x_ref[i], 
            obs_C1R1=C1R1_ref[i])
            )(
            x_mean=x_aux, 
            Mean=Mean, 
            Col_Cov=Col_Cov, 
            Row_Scale=Row_Scale, 
            df=df
        )
        log_weights_ancestor = log_weights_aux + l_x
        weights_ancestor = np.exp(log_weights_ancestor - np.max(log_weights_ancestor))
        weights_ancestor /= np.sum(weights_ancestor)
        
        # sample an ancestor index for reference trajectory
        ref_idx = np.searchsorted(np.cumsum(weights_ancestor), rng.random())
        
        # set ancestor index
        idx[-1] = ref_idx
        
        # save genealogy
        ancestor_idx[i-1] = idx
        
        
        
        ### Step 3: Make a proposal by generating samples from the hirachical 
        # model
        
        # sample from proposal for x at time t
        Sigma_X[i] = jax.vmap(
            functools.partial(f_x, I=ctrl_input[i-1], dt=dt)
            )(
            x=Sigma_X[i-1,idx],
            C_1=Sigma_C1R1[i-1,idx,0],
            R_1=Sigma_C1R1[i-1,idx,1]
        ) + w((N_particles,)).flatten()
        
        # set reference trajectory for state x
        Sigma_X[i,-1] = x_ref[i]
        
        ## sample proposal for C1 and R1
        # evaluate basis functions
        basis = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i])
        
        # calculate predictive distribution for C1 and R1
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
        Sigma_C1R1[i,-1] = C1R1_ref[i]
        
        
        
        ### Step 4: Update reference statistic
        basis_ref = basis_fcn(x_ref[i])
        T_0, T_1, T_2, T_3 = prior_mniw_calcStatistics(
            C1R1_ref[i],
            basis_ref
        )
        GP_stats_ref[0] -= T_0
        GP_stats_ref[1] -= T_1
        GP_stats_ref[2] -= T_2
        GP_stats_ref[3] -= T_3
        
        
        
        ### Step 5: Update hyperparameters
        T_0, T_1, T_2, T_3 = jax.vmap(prior_mniw_calcStatistics)(
            Sigma_C1R1[i],
            basis
        )
        GP_stats_ancestor[0] = GP_stats_ancestor[0][idx] + T_0
        GP_stats_ancestor[1] = GP_stats_ancestor[1][idx] + T_1
        GP_stats_ancestor[2] = GP_stats_ancestor[2][idx] + T_2
        GP_stats_ancestor[3] = GP_stats_ancestor[3][idx] + T_3
        
            
        
        ### Step 6: Calculate new weights
        Sigma_Y[i] = jax.vmap(functools.partial(f_y, I=ctrl_input[i]))(x=Sigma_X[i])
        l_fy = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(Sigma_Y[i])
        
        l_C0_clip = np.all(np.vstack([cl_C1 >= Sigma_C1R1[i,:,0], Sigma_C1R1[i,:,0] >= cu_C1]), axis=0)
        l_C0_clip = np.array(l_C0_clip, dtype=np.float_) * -500
        l_R0_clip = np.all(np.vstack([cl_R1 >= Sigma_C1R1[i,:,1], Sigma_C1R1[i,:,1] >= cu_R1]), axis=0)
        l_R0_clip = np.array(l_R0_clip, dtype=np.float_) * -500
        
        log_weights[i] = l_fy + l_C0_clip + l_R0_clip - l_fy_aux[idx] - l_C0_clip_aux[idx] - l_R0_clip_aux[idx]
        
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
    C1R1_traj = reconstruct_trajectory(Sigma_C1R1, ancestor_idx, idx_traj)
    y_traj = reconstruct_trajectory(Sigma_Y, ancestor_idx, idx_traj)
    
    return x_traj, C1R1_traj, y_traj