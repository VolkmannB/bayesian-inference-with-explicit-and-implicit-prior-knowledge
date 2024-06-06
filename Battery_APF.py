import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from src.Battery import BatterySSM, basis_fcn, default_para, z_ip
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_Predictive
from src.BayesianInferrence import prior_mniw_updateStatistics, prior_mniw_CondPredictive
from src.Filtering import squared_error, systematic_SISR



################################################################################
# Model


# sim para
N = 100
y1_var=5e-2



### Define GP prior and variables for sufficient statistics

# parameter limits
l_C1, u_C1 = 2, 20
l_R1, u_R1 = 5, 10

# model prior C1, R1
N_z = z_ip.shape[0]
GP_prior_C1R1 = list(prior_mniw_2naturalPara(
    np.zeros((2, N_z)),
    np.eye(N_z)*10,
    np.diag([1, 1]),
    0
))
GP_stats_C1R1 = [
    np.zeros((N, N_z, 2)),
    np.zeros((N, N_z, N_z)),
    np.zeros((N, 2, 2)),
    np.zeros((N,))
]



### Load data
time_gap_thresh = pd.Timedelta("1h")
data = pd.read_csv("./src/Measurements/Everlast_35E_003.csv", sep=",")
data["Time"] = pd.to_datetime(data["Time"])
data = data.set_index("Time")

# segment data
data["Block"] = (data.index.to_series().diff() > time_gap_thresh).cumsum()
blocks = [group for _, group in data.groupby("Block")]
data = blocks[0]
del blocks

# resampling to uniform time steps
dt = pd.Timedelta("1s")
data = data.resample("50ms", origin="start").interpolate().resample("1s", origin="start").asfreq()

data = data.iloc[:20000]
steps = data.shape[0]

# init model
model = BatterySSM(**default_para)



# initial system state
x0 = np.array([data["Voltage"].iloc[0]-default_para["V_0"]])
P0 = np.diag([5e-3])
# np.random.seed(573573)

# process and measurement noise
R = np.diag([0.001])
Q = np.diag([5e-6])
w = lambda n=1: np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((R.shape[0],)), R, n)



################################################################################
# Simulation

# time series for plot
Sigma_X = np.zeros((steps,N))
Sigma_Y = np.zeros((steps,N))
Sigma_C1R1 = np.zeros((steps,N,2))

Mean_C1R1 = np.zeros((steps, 2, N_z))
Col_Cov_C1R1 = np.zeros((steps, N_z, N_z))
Row_Scale_C1R1 = np.zeros((steps, 2, 2))
df_C1R1 = np.zeros((steps,))

# input
ctrl_input = data["Current"].to_numpy()

# measurments
Y = data[["Voltage"]].to_numpy()



### Set all initial values

# initial values for states
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,)).flatten()
Sigma_C1R1[0,:,0] = np.random.uniform(l_C1, u_C1, (N,))
Sigma_C1R1[0,:,1] = np.random.uniform(l_R1, u_R1, (N,))
weights = np.ones((steps,N))/N

# update GP
phi_0 = jax.vmap(
    functools.partial(basis_fcn)
    )(Sigma_X[0])
GP_stats_C1R1 = list(jax.vmap(prior_mniw_updateStatistics)(
    *GP_stats_C1R1,
    Sigma_C1R1[0,...],
    phi_0
))


# simulation loop
for i in tqdm(range(1, steps), desc="Running simulation"):
    
    ### Step 1: Propagate GP parameters in time
    dt = (data.index[i] - data.index[i-1]).seconds
    
    # apply forgetting operator to statistics for t-1 -> t
    GP_stats_C1R1[0] *= 1 - 1/1e3
    GP_stats_C1R1[1] *= 1 - 1/1e3
    GP_stats_C1R1[2] *= 1 - 1/1e3
    GP_stats_C1R1[3] *= 1 - 1/1e3
        
    # calculate parameters of GP from prior and sufficient statistics
    GP_para_C1R1 = list(jax.vmap(prior_mniw_2naturalPara_inv)(
        GP_prior_C1R1[0] + GP_stats_C1R1[0],
        GP_prior_C1R1[1] + GP_stats_C1R1[1],
        GP_prior_C1R1[2] + GP_stats_C1R1[2],
        GP_prior_C1R1[3] + GP_stats_C1R1[3]
    ))
    
    
    
    ### Step 2: According to the algorithm of the auxiliary PF, resample 
    # particles according to the first stage weights
    
    # create auxiliary variable for state x
    x_aux = jax.vmap(
        functools.partial(model, I=ctrl_input[i-1], dt=dt))(
            x=Sigma_X[i-1],
            C_1=Sigma_C1R1[i-1,...,0]*1e2,
            R_1=Sigma_C1R1[i-1,...,1]*1e3
        )
    
    # create auxiliary variable for C1, R1 and R0
    phi_0 = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i-1])
    phi_1 = jax.vmap(functools.partial(basis_fcn))(x_aux)
    C1R1_aux = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=y1_var)
        )(
            mean=GP_para_C1R1[0],
            col_cov=GP_para_C1R1[1],
            row_scale=GP_para_C1R1[2],
            df=GP_para_C1R1[3],
            y1=Sigma_C1R1[i-1,...],
            basis1=phi_0,
            basis2=phi_1
    )[0]
    
    # calculate first stage weights
    y_aux = jax.vmap(
        functools.partial(model.fy, I=ctrl_input[i], R_0=23e-3)
        )(x=x_aux)
    l_fy = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(x=y_aux)
    l_C0_clip = np.all(np.vstack([l_C1 <= C1R1_aux[:,0], C1R1_aux[:,0] <= u_C1]), axis=0)
    l_R0_clip = np.all(np.vstack([l_R1 <= C1R1_aux[:,1], C1R1_aux[:,1] <= u_R1]), axis=0)
    p = weights[i-1] * l_fy * l_C0_clip * l_R0_clip
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
    GP_stats_C1R1[0] = GP_stats_C1R1[0][idx,...]
    GP_stats_C1R1[1] = GP_stats_C1R1[1][idx,...]
    GP_stats_C1R1[2] = GP_stats_C1R1[2][idx,...]
    GP_stats_C1R1[3] = GP_stats_C1R1[3][idx,...]
    GP_para_C1R1[0] = GP_para_C1R1[0][idx,...]
    GP_para_C1R1[1] = GP_para_C1R1[1][idx,...]
    GP_para_C1R1[2] = GP_para_C1R1[2][idx,...]
    GP_para_C1R1[3] = GP_para_C1R1[3][idx,...]
    
    Sigma_X[i-1] = Sigma_X[i-1,idx,...]
    Sigma_C1R1[i-1] = Sigma_C1R1[i-1,idx,...]
    
    
    
    ### Step 3: Make a proposal by generating samples from the hirachical 
    # model
    
    # sample from proposal for x at time t
    w_x = w((N,)).flatten()
    Sigma_X[i] = jax.vmap(
        functools.partial(model, I=ctrl_input[i-1], dt=dt))(
            x=Sigma_X[i-1],
            C_1=Sigma_C1R1[i-1,:,0]*1e2,
            R_1=Sigma_C1R1[i-1,:,1]*1e3
        ) + w_x
    
    ## sample proposal for alpha and beta at time t
    # evaluate basis functions
    phi_0 = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i-1])
    phi_1 = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i])
    
    # calculate conditional predictive distribution for C1 and R1
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=y1_var)
        )(
            mean=GP_para_C1R1[0],
            col_cov=GP_para_C1R1[1],
            row_scale=GP_para_C1R1[2],
            df=GP_para_C1R1[3],
            y1=Sigma_C1R1[i-1,...],
            basis1=phi_0,
            basis2=phi_1
    )
    
    # generate samples
    c_col_scale_chol = np.linalg.cholesky(c_col_scale)
    c_row_scale_chol = np.linalg.cholesky(c_row_scale)
    t_samples = np.random.standard_t(df=df, size=(1,2,N)).T
    Sigma_C1R1[i] = c_mean + np.squeeze(np.einsum(
            '...ij,...jk,...kf->...if', 
            c_row_scale_chol, 
            t_samples, 
            c_col_scale_chol
        ))
        
        
    
    # Update the sufficient statistics of GP with new proposal
    GP_stats_C1R1 = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_stats_C1R1,
        Sigma_C1R1[i],
        phi_1
    ))
    
    # calculate new weights
    Sigma_Y[i] = jax.vmap(
        functools.partial(model.fy, I=ctrl_input[i], R_0=23e-3)
        )(x=Sigma_X[i])
    q_fy = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(Sigma_Y[i])
    q_C0_clip = np.all(np.vstack([l_C1 <= Sigma_C1R1[i,:,0], Sigma_C1R1[i,:,0] <= u_C1]), axis=0)
    q_R0_clip = np.all(np.vstack([l_R1 <= Sigma_C1R1[i,:,1], Sigma_C1R1[i,:,1] <= u_R1]), axis=0)
    weights[i] = q_fy * q_C0_clip * q_R0_clip / p[idx]
    weights[i] = weights[i]/np.sum(weights[i])
    
    
    
    # logging
    GP_para_logging_C1R1 = prior_mniw_2naturalPara_inv(
        GP_prior_C1R1[0] + np.einsum('n...,n->...', GP_stats_C1R1[0], weights[i]),
        GP_prior_C1R1[1] + np.einsum('n...,n->...', GP_stats_C1R1[1], weights[i]),
        GP_prior_C1R1[2] + np.einsum('n...,n->...', GP_stats_C1R1[2], weights[i]),
        GP_prior_C1R1[3] + np.einsum('n...,n->...', GP_stats_C1R1[3], weights[i])
    )
    Mean_C1R1[i] = GP_para_logging_C1R1[0]
    Col_Cov_C1R1[i] = GP_para_logging_C1R1[1]
    Row_Scale_C1R1[i] = GP_para_logging_C1R1[2]
    df_C1R1[i] = GP_para_logging_C1R1[3]
    
    #abort
    if np.any(np.isnan(weights[i])):
        print("Particle degeneration at new weights")
        break



################################################################################
# Plotting

X_pred = np.einsum('...n,...n->...', Sigma_X, weights)
Y_pred = np.einsum('...n,...n->...', Sigma_Y, weights)



# plot data
fig_data, ax_data = plt.subplots(2,1, layout='tight')

ax_data[0].plot(data.index, data["Voltage"], color='blue', label='Meas')
ax_data[0].plot(data.index[1:], Y_pred[1:], color='orange', label='Pred')

ax_data[1].plot(data.index, data["Current"], color='blue', label='Meas')



# plot learned function
fig_C1R1, ax_C1R1 = plt.subplots(2,6, layout='tight')
V = jnp.linspace(0, 2.4, 500)
phi = jax.vmap(basis_fcn)(V[...,None])

# label
ax_C1R1[0,0].set_ylabel('Capacity in F x1e2')
ax_C1R1[1,0].set_ylabel('Resistance in Ohm x1e3')

# function
for i in range(6):
    ax_C1R1[0,i].set_ylim(0, 20)
    ax_C1R1[1,i].set_ylim(0, 15)
    
    mean, col_scale, row_scale, df = prior_mniw_Predictive(
        mean=Mean_C1R1[int((i+1)/6*steps)-1], 
        col_cov=Col_Cov_C1R1[int((i+1)/6*steps)-1], 
        row_scale=Row_Scale_C1R1[int((i+1)/6*steps)-1], 
        df=df_C1R1[int((i+1)/6*steps)-1], 
        basis=phi
        )
    std0 = np.diag(np.linalg.cholesky(col_scale*row_scale[0,0]))
    std1 = np.diag(np.linalg.cholesky(col_scale*row_scale[1,1]))
    
    ax_C1R1[0,i].plot(V, mean[:,0], color='blue')
    ax_C1R1[0,i].fill_between(V, mean[:,0]-std0, mean[:,0]+std0, color='blue', alpha=0.2)
    
    ax_C1R1[1,i].plot(V, mean[:,1], color='blue')
    ax_C1R1[1,i].fill_between(V, mean[:,1]-std1, mean[:,1]+std1, color='blue', alpha=0.2)



plt.show()