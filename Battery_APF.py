import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


from src.Battery import BatterySSM, features_R, features_V, default_para, z_ip, zI_ip
from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_updateStatistics, prior_mniw_CondPredictive
from src.Filtering import squared_error, systematic_SISR



################################################################################
# Model


# sim para
N = 200
Q_cap = 3.4



### Define GP prior and variables for sufficient statistics

# model prior alpha, beta
N_z = z_ip.shape[0]
GP_prior_V = list(prior_mniw_2naturalPara(
    np.diag([5e-2, 2e-3]) @ np.ones((2, N_z)),
    np.eye(N_z)*1e-3,
    np.diag([1e-8, 1e-8]),
    0
))

# model prior resistor
N_zI = zI_ip.shape[0]
GP_prior_R = list(prior_mniw_2naturalPara(
    np.ones((1, N_zI))*33e-3,
    np.eye(N_zI)*1e-3,
    np.eye(1)*1e-8,
    0
))

# parameters for the sufficient statistics
GP_stats_V = [
    np.zeros((N, N_z, 2)),
    np.zeros((N, N_z, N_z)),
    np.zeros((N, 2, 2)),
    np.zeros((N,))
]
GP_stats_R = [
    np.zeros((N, N_zI, 1)),
    np.zeros((N, N_zI, N_zI)),
    np.zeros((N, 1, 1)),
    np.zeros((N,))
]



### Load data
data = pd.read_csv("./src/Measurements/Everlast_35E_002.csv", sep=",")
data["Time"] = pd.to_datetime(data["Time"])
data = data.set_index("Time")
# dt = (data.index[1] - data.index[0]).seconds
# data = data.resample(rule=dt, origin='start').asfreq()#.interpolate(method='cubicspline')
steps = 10000#data.shape[0]
data = data.iloc[:steps]

# init model
default_para = {
    'R_c': 0.407, # from literature
    'C_c': 43.5, # from literature
    'T_amb': 27, # given
    'V_0': 2.56
}
model = BatterySSM(**default_para)



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
data_fig.update_layout(showlegend=False)
data_fig.update_yaxes(title_text='Voltage in [V]', row=1, col=1)
data_fig.update_yaxes(title_text='Curren in [A]', row=2, col=1)
data_fig.update_yaxes(title_text='Temperature in [C]', row=3, col=1)
data_fig.update_xaxes(title_text='Time', row=3, col=1)
data_fig.update_layout(font_size=24)



# initial system state
x0 = np.array([0.9, data["Voltage"][0]-default_para["V_0"], data["Temperature"][0]])
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
Sigma_V = np.zeros((steps,N,2))
Sigma_R = np.zeros((steps,N))

A_V = np.zeros((steps, 2, N_z))
A_R = np.zeros((steps, N_zI))

# input
ctrl_input = data["Current"].to_numpy()

# measurments
Y = data[["Voltage", "Temperature"]].to_numpy()


### Set all initial values

# initial values for states
l_V, u_V = np.array([2e-2, 0.5e-3]), np.array([8e-2, 3.5e-3])
l_R, u_R = 25e-3, 50e-3
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
Sigma_V[0,...] = np.random.uniform(l_V, u_V, (N,2))#np.random.multivariate_normal([0.01, 0.0007], np.diag([0.06**2, 0.00074**2]), (N,))
Sigma_R[0,...] = np.random.uniform(l_R, u_R, (N,))#np.random.normal(0.04, 0.088**2, (N,))
weights = np.ones((steps,N))/N

# update GP
phi_V = jax.vmap(
    functools.partial(features_V)
    )(Sigma_X[0])
phi_R = jax.vmap(
    functools.partial(features_R, I=ctrl_input[0])
    )(Sigma_X[0])
GP_stats_V = list(jax.vmap(prior_mniw_updateStatistics)(
    *GP_stats_V,
    Sigma_V[0,...],
    phi_V
))
GP_stats_R = list(jax.vmap(prior_mniw_updateStatistics)(
    *GP_stats_R,
    Sigma_R[0,...],
    phi_R
))


# simulation loop
for i in tqdm(range(1, steps), desc="Running simulation"):
    
    ### Step 1: Propagate GP parameters in time
    dt = (data.index[i] - data.index[i-1]).seconds
    
    # apply forgetting operator to statistics for t-1 -> t
    GP_stats_V[0] *= 0.999
    GP_stats_V[1] *= 0.999
    GP_stats_V[2] *= 0.999
    GP_stats_V[3] *= 0.999
    GP_stats_R[0] *= 0.999
    GP_stats_R[1] *= 0.999
    GP_stats_R[2] *= 0.999
    GP_stats_R[3] *= 0.999
        
    # calculate parameters of GP from prior and sufficient statistics
    GP_para_V = list(jax.vmap(prior_mniw_2naturalPara_inv)(
        GP_prior_V[0] + GP_stats_V[0],
        GP_prior_V[1] + GP_stats_V[1],
        GP_prior_V[2] + GP_stats_V[2],
        GP_prior_V[3] + GP_stats_V[3]
    ))
    GP_para_R = list(jax.vmap(prior_mniw_2naturalPara_inv)(
        GP_prior_R[0] + GP_stats_R[0],
        GP_prior_R[1] + GP_stats_R[1],
        GP_prior_R[2] + GP_stats_R[2],
        GP_prior_R[3] + GP_stats_R[3]
    ))
    
    
    
    ### Step 2: According to the algorithm of the auxiliary PF, resample 
    # particles according to the first stage weights
    
    # create auxiliary variable for state x
    x_aux = jax.vmap(
        functools.partial(model, I=ctrl_input[i-1], Q=Q_cap, dt=dt))(
            x=Sigma_X[i-1],
            alpha=Sigma_V[i-1,...,0],
            beta=Sigma_V[i-1,...,1],
            R_0=Sigma_R[i-1]
        )
    
    # create auxiliary variable for resistance R
    phi_R0 = jax.vmap(functools.partial(features_R, I=ctrl_input[i-1]))(Sigma_X[i-1])
    phi_R1 = jax.vmap(functools.partial(features_R, I=ctrl_input[i]))(x_aux)
    R_aux = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-1)
        )(
            mean=GP_para_R[0],
            col_cov=GP_para_R[1],
            row_scale=GP_para_R[2],
            df=GP_para_R[3],
            y1=Sigma_R[i-1,...],
            basis1=phi_R0,
            basis2=phi_R1
    )[0]
    
    # create auxiliary variable for alpha and beta
    phi_V0 = jax.vmap(functools.partial(features_V))(Sigma_X[i-1])
    phi_V1 = jax.vmap(functools.partial(features_V))(x_aux)
    V_aux = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-1)
        )(
            mean=GP_para_V[0],
            col_cov=GP_para_V[1],
            row_scale=GP_para_V[2],
            df=GP_para_V[3],
            y1=Sigma_V[i-1,...],
            basis1=phi_V0,
            basis2=phi_V1
    )[0]
    
    # calculate first stage weights
    y_aux = jax.vmap(
        functools.partial(model.fy, I=ctrl_input[i])
        )(
            x=x_aux,
            R_0=R_aux)
    l_fy = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(x=y_aux)
    l_clipR = np.all([0 <= R_aux, R_aux <= 1e-1], axis=0)
    l_clipV = np.all(np.hstack([0 <= V_aux, V_aux <= 1e-1]), axis=1)
    p = weights[i-1] * l_fy * l_clipR * l_clipV
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
    GP_stats_V[0] = GP_stats_V[0][idx,...]
    GP_stats_V[1] = GP_stats_V[1][idx,...]
    GP_stats_V[2] = GP_stats_V[2][idx,...]
    GP_stats_V[3] = GP_stats_V[3][idx,...]
    GP_para_V[0] = GP_para_V[0][idx,...]
    GP_para_V[1] = GP_para_V[1][idx,...]
    GP_para_V[2] = GP_para_V[2][idx,...]
    GP_para_V[3] = GP_para_V[3][idx,...]
    
    GP_stats_R[0] = GP_stats_R[0][idx,...]
    GP_stats_R[1] = GP_stats_R[1][idx,...]
    GP_stats_R[2] = GP_stats_R[2][idx,...]
    GP_stats_R[3] = GP_stats_R[3][idx,...]
    GP_para_R[0] = GP_para_R[0][idx,...]
    GP_para_R[1] = GP_para_R[1][idx,...]
    GP_para_R[2] = GP_para_R[2][idx,...]
    GP_para_R[3] = GP_para_R[3][idx,...]
    
    Sigma_X[i-1] = Sigma_X[i-1,idx,...]
    Sigma_V[i-1] = Sigma_V[i-1,idx,...]
    Sigma_R[i-1] = Sigma_R[i-1,idx]
    
    
    
    ### Step 3: Make a proposal by generating samples from the hirachical 
    # model
    
    # sample from proposal for x at time t
    w_x = w((N,))
    Sigma_X[i] = jax.vmap(
        functools.partial(model, I=ctrl_input[i-1], Q=Q_cap, dt=dt))(
            x=Sigma_X[i-1],
            alpha=Sigma_V[i-1,...,0],
            beta=Sigma_V[i-1,...,1],
            R_0=Sigma_R[i-1]
        ) + w_x
    
    ## sample proposal for alpha and beta at time t
    # evaluate basis functions
    phi_V0 = jax.vmap(functools.partial(features_V))(Sigma_X[i-1])
    phi_V1 = jax.vmap(functools.partial(features_V))(Sigma_X[i])
    
    # calculate conditional predictive distribution
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-1)
        )(
            mean=GP_para_V[0],
            col_cov=GP_para_V[1],
            row_scale=GP_para_V[2],
            df=GP_para_V[3],
            y1=Sigma_V[i-1,...],
            basis1=phi_V0,
            basis2=phi_V1
    )
    
    # generate samples
    c_col_scale_chol = np.linalg.cholesky(c_col_scale)
    c_row_scale_chol = np.linalg.cholesky(c_row_scale)
    t_samples = np.random.standard_t(df=df, size=(2,200)).T
    Sigma_V[i] = c_mean + np.einsum(
        '...ij,...j,...jk->...k', 
        c_col_scale_chol, 
        t_samples, 
        c_row_scale_chol
        )
    
    ## sample proposal for R at time t
    # evaluate basis functions
    phi_R0 = jax.vmap(functools.partial(features_R, I=ctrl_input[i]))(Sigma_X[i-1])
    phi_R1 = jax.vmap(functools.partial(features_R, I=ctrl_input[i]))(Sigma_X[i])
    
    # calculate conditional predictive distribution
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=3e-1)
        )(
            mean=GP_para_R[0],
            col_cov=GP_para_R[1],
            row_scale=GP_para_R[2],
            df=GP_para_R[3],
            y1=Sigma_R[i-1,...],
            basis1=phi_R0,
            basis2=phi_R1
    )
    
    # generate samples
    c_col_scale_chol = np.sqrt(np.squeeze(c_col_scale))
    c_row_scale_chol = np.sqrt(np.squeeze(c_row_scale))
    t_samples = np.random.standard_t(df=df)
    Sigma_R[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
        
        
    
    # Update the sufficient statistics of GP with new proposal
    GP_stats_V = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_stats_V,
        Sigma_V[i],
        phi_V1
    ))
    GP_stats_R = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_stats_R,
        Sigma_R[i],
        phi_R1
    ))
    
    # calculate new weights
    Sigma_Y[i] = jax.vmap(
        functools.partial(model.fy, I=ctrl_input[i])
        )(
            x=Sigma_X[i],
            R_0=Sigma_R[i])
    q_fy = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(Sigma_Y[i])
    q_clipR = np.all([l_R <= Sigma_R[i], Sigma_R[i] <= u_R], axis=0)
    q_clipV = np.all(np.hstack([l_V <= Sigma_V[i], Sigma_V[i] <= u_V]), axis=1)
    weights[i] = q_fy * q_clipR * q_clipV / p[idx]
    weights[i] = weights[i]/np.sum(weights[i])
    
    
    
    # logging
    GP_para_logging_V = prior_mniw_2naturalPara_inv(
        GP_prior_V[0] + np.einsum('n...,n->...', GP_stats_V[0], weights[i]),
        GP_prior_V[1] + np.einsum('n...,n->...', GP_stats_V[1], weights[i]),
        GP_prior_V[2] + np.einsum('n...,n->...', GP_stats_V[2], weights[i]),
        GP_prior_V[3] + np.einsum('n...,n->...', GP_stats_V[3], weights[i])
    )
    GP_para_logging_R = prior_mniw_2naturalPara_inv(
        GP_prior_R[0] + np.einsum('n...,n->...', GP_stats_R[0], weights[i]),
        GP_prior_R[1] + np.einsum('n...,n->...', GP_stats_R[1], weights[i]),
        GP_prior_R[2] + np.einsum('n...,n->...', GP_stats_R[2], weights[i]),
        GP_prior_R[3] + np.einsum('n...,n->...', GP_stats_R[3], weights[i])
    )
    A_V[i] = GP_para_logging_V[0]
    A_R[i] = GP_para_logging_R[0]
    
    #abort
    if np.any(np.isnan(weights[i])):
        print("Particle degeneration at new weights")
        break



################################################################################
# Plotting


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
data_fig.show()