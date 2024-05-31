import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


from src.Battery import BatterySSM, basis_fcn, default_para, z_ip
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_updateStatistics, prior_mniw_CondPredictive
from src.Filtering import squared_error, systematic_SISR



################################################################################
# Model


# sim para
N = 300
Q_cap = 3.4



### Define GP prior and variables for sufficient statistics

# parameter limits
l_CRR, u_CRR = np.array([0.2e3, 5e-3, 20e-3]), np.array([2.5e3, 0.25, 60e-3])

# model prior C1, R1, R0
N_z = z_ip.shape[0]
GP_prior = list(prior_mniw_2naturalPara(
    np.diag((l_CRR+u_CRR)/2) @ np.ones((3, N_z)),
    np.eye(N_z),
    np.diag([1e1, 1e-3, 1e-4]),
    0
))

# parameters for the sufficient statistics
GP_stats = [
    np.zeros((N, N_z, 3)),
    np.zeros((N, N_z, N_z)),
    np.zeros((N, 3, 3)),
    np.zeros((N,))
]



### Load data
data = pd.read_csv("./src/Measurements/Everlast_35E_002.csv", sep=",")
data["Time"] = pd.to_datetime(data["Time"])
data = data.set_index("Time")
start=360
end=10000
data = data.iloc[368:end]
steps = data.shape[0]

# init model
model = BatterySSM(**default_para)



# initial system state
x0 = np.array([0.9, data["Voltage"].iloc[0]-default_para["V_0"], data["Temperature"].iloc[0]])
P0 = np.diag([1e-2, 1e-2, 1e-2])
np.random.seed(573573)

# process and measurement noise
R = np.diag([0.001, 0.25])
Q = np.diag([1e-12, 5e-6, 1e-4])
w = lambda n=1: np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((R.shape[0],)), R, n)



################################################################################
# Simulation

# time series for plot

Sigma_X = np.zeros((steps,N,3))
Sigma_Y = np.zeros((steps,N,2))
Sigma_CRR = np.zeros((steps,N,3))

A_CRR = np.zeros((steps, 3, N_z))

# input
ctrl_input = data["Current"].to_numpy()

# measurments
Y = data[["Voltage", "Temperature"]].to_numpy()


### Set all initial values

# initial values for states
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
Sigma_CRR[0,...] = np.random.uniform(l_CRR, u_CRR, (N,3))
weights = np.ones((steps,N))/N

# update GP
phi_0 = jax.vmap(
    functools.partial(basis_fcn)
    )(Sigma_X[0])
GP_stats = list(jax.vmap(prior_mniw_updateStatistics)(
    *GP_stats,
    Sigma_CRR[0,...],
    phi_0
))


# simulation loop
for i in tqdm(range(1, steps), desc="Running simulation"):
    
    ### Step 1: Propagate GP parameters in time
    dt = (data.index[i] - data.index[i-1]).seconds
    
    # apply forgetting operator to statistics for t-1 -> t
    GP_stats[0] *= 0.999
    GP_stats[1] *= 0.999
    GP_stats[2] *= 0.999
    GP_stats[3] *= 0.999
        
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
        functools.partial(model, I=ctrl_input[i-1], dt=dt))(
            x=Sigma_X[i-1],
            C_1=Sigma_CRR[i-1,...,0],
            R_1=Sigma_CRR[i-1,...,1],
            R_0=Sigma_CRR[i-1,...,2]
        )
    
    # create auxiliary variable for alpha and beta
    phi_0 = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i-1])
    phi_1 = jax.vmap(functools.partial(basis_fcn))(x_aux)
    CRR_aux = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=2e-2)
        )(
            mean=GP_para[0],
            col_cov=GP_para[1],
            row_scale=GP_para[2],
            df=GP_para[3],
            y1=Sigma_CRR[i-1,...],
            basis1=phi_0,
            basis2=phi_1
    )[0]
    
    # calculate first stage weights
    y_aux = jax.vmap(
        functools.partial(model.fy, I=ctrl_input[i])
        )(
            x=x_aux,
            R_0=CRR_aux[:,2])
    l_fy = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(x=y_aux)
    l_clip = np.all(np.hstack([l_CRR <= CRR_aux, CRR_aux <= u_CRR]), axis=1)
    p = weights[i-1] * l_fy * l_clip
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
    GP_stats[0] = GP_stats[0][idx,...]
    GP_stats[1] = GP_stats[1][idx,...]
    GP_stats[2] = GP_stats[2][idx,...]
    GP_stats[3] = GP_stats[3][idx,...]
    GP_para[0] = GP_para[0][idx,...]
    GP_para[1] = GP_para[1][idx,...]
    GP_para[2] = GP_para[2][idx,...]
    GP_para[3] = GP_para[3][idx,...]
    
    Sigma_X[i-1] = Sigma_X[i-1,idx,...]
    Sigma_CRR[i-1] = Sigma_CRR[i-1,idx,...]
    
    
    
    ### Step 3: Make a proposal by generating samples from the hirachical 
    # model
    
    # sample from proposal for x at time t
    w_x = w((N,))
    Sigma_X[i] = jax.vmap(
        functools.partial(model, I=ctrl_input[i-1], dt=dt))(
            x=Sigma_X[i-1],
            C_1=Sigma_CRR[i-1,:,0],
            R_1=Sigma_CRR[i-1,:,1],
            R_0=Sigma_CRR[i-1,:,2]
        ) + w_x
    
    ## sample proposal for alpha and beta at time t
    # evaluate basis functions
    phi_0 = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i-1])
    phi_1 = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i])
    
    # calculate conditional predictive distribution
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-1)
        )(
            mean=GP_para[0],
            col_cov=GP_para[1],
            row_scale=GP_para[2],
            df=GP_para[3],
            y1=Sigma_CRR[i-1,...],
            basis1=phi_0,
            basis2=phi_1
    )
    
    # generate samples
    c_col_scale_chol = np.linalg.cholesky(c_col_scale)
    c_row_scale_chol = np.linalg.cholesky(c_row_scale)
    t_samples = np.random.standard_t(df=df, size=(3,N)).T
    Sigma_CRR[i] = c_mean + np.einsum(
        '...ij,...j,...jk->...k', 
        c_col_scale_chol, 
        t_samples, 
        c_row_scale_chol
        )
        
        
    
    # Update the sufficient statistics of GP with new proposal
    GP_stats = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_stats,
        Sigma_CRR[i],
        phi_1
    ))
    
    # calculate new weights
    Sigma_Y[i] = jax.vmap(
        functools.partial(model.fy, I=ctrl_input[i])
        )(
            x=Sigma_X[i],
            R_0=Sigma_CRR[i,:,2])
    q_fy = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(Sigma_Y[i])
    q_clip = np.all(np.hstack([l_CRR <= Sigma_CRR[i], Sigma_CRR[i] <= u_CRR]), axis=1)
    weights[i] = q_fy * q_clip / p[idx]
    weights[i] = weights[i]/np.sum(weights[i])
    
    
    
    # logging
    GP_para_logging_CCR = prior_mniw_2naturalPara_inv(
        GP_prior[0] + np.einsum('n...,n->...', GP_stats[0], weights[i]),
        GP_prior[1] + np.einsum('n...,n->...', GP_stats[1], weights[i]),
        GP_prior[2] + np.einsum('n...,n->...', GP_stats[2], weights[i]),
        GP_prior[3] + np.einsum('n...,n->...', GP_stats[3], weights[i])
    )
    A_CRR[i] = GP_para_logging_CCR[0]
    
    #abort
    if np.any(np.isnan(weights[i])):
        print("Particle degeneration at new weights")
        break



################################################################################
# Plotting

### give a plot of the data
data_fig = make_subplots(3,1)
data_fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Voltage"],
        mode='markers',
        marker=dict(
            color='blue',
            size=4
        )
    ),
    row=1,
    col=1
)
data_fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Current"],
        mode='markers',
        marker=dict(
            color='blue',
            size=4
        )
    ),
    row=2,
    col=1
)
data_fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Temperature"],
        mode='markers',
        marker=dict(
            color='blue',
            size=4
        )
    ),
    row=3,
    col=1
)


# plot predicted observations
X_pred = np.einsum('...ni,...n->...i', Sigma_X, weights)
Y_pred = np.einsum('...ni,...n->...i', Sigma_Y, weights)
data_fig.add_trace(
    go.Scatter(
        x=data.index[1:],
        y=Y_pred[1:,0],
        mode='lines',
        line=dict(
            color='orange',
            width=2
        )
    ),
    row=1,
    col=1
)
data_fig.add_trace(
    go.Scatter(
        x=data.index[1:],
        y=Y_pred[1:,1],
        mode='lines',
        line=dict(
            color='orange',
            width=2
        )
    ),
    row=3,
    col=1
)
data_fig.update_layout(showlegend=False)
data_fig.update_yaxes(title_text='Voltage in [V]', row=1, col=1)
data_fig.update_yaxes(title_text='Curren in [A]', row=2, col=1)
data_fig.update_yaxes(title_text='Temperature in [C]', row=3, col=1)
data_fig.update_xaxes(title_text='Time', row=3, col=1)
data_fig.update_layout(font_size=24)
data_fig.show()