import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import pandas as pd

import matplotlib.pyplot as plt

from src.Battery import BatterySSM, basis_fcn, sd, default_para, z_ip
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_Predictive
from src.BayesianInferrence import prior_mniw_updateStatistics
from src.BayesianInferrence import prior_mniw_CondPredictive
from src.Filtering import squared_error, systematic_SISR
from src.Publication_Plotting import plot_BFE_1D, generate_BFE_TimeSlices
from src.Publication_Plotting import apply_basic_formatting, plot_Data



################################################################################
# Model


# sim para
N = 100
forget_factor = 1 - 1/1e3
y1_var=5e-2



### Define GP prior and variables for sufficient statistics

# parameter limits
l_C1, u_C1 = 2, 20
l_R1, u_R1 = 5, 10

# domain of BFE
offset_C1 = (u_C1 - l_C1)/2
offset_R1 = (u_R1 - l_R1)/2
cl_C1, cu_C1 = l_C1 - offset_C1, u_C1 - offset_C1
cl_R1, cu_R1 = l_R1 - offset_R1, u_R1 - offset_R1

# model prior C1, R1
N_z = z_ip.shape[0]
GP_prior_C1R1 = list(prior_mniw_2naturalPara(
    np.zeros((2, N_z)),
    np.diag(sd),
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
data = blocks[3]
del blocks

# resampling to uniform time steps
dt = pd.Timedelta("1s")
data = data.resample("5ms", origin="start").interpolate().resample("1s", origin="start").asfreq()

# data = data.iloc[:2000]
steps = data.shape[0]

# init model
model = BatterySSM(**default_para)



# initial system state
x0 = np.array([data["Voltage"].iloc[0]-default_para["V_0"]])
P0 = np.diag([5e-3])
np.random.seed(16723573)

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
Sigma_C1R1[0,:,0] = np.random.uniform(cl_C1, cu_C1, (N,))
Sigma_C1R1[0,:,1] = np.random.uniform(cl_R1, cu_R1, (N,))
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
    GP_stats_C1R1[0] *= forget_factor
    GP_stats_C1R1[1] *= forget_factor
    GP_stats_C1R1[2] *= forget_factor
    GP_stats_C1R1[3] *= forget_factor
        
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
            C_1=(offset_C1+Sigma_C1R1[i-1,...,0])*1e2,
            R_1=(offset_R1+Sigma_C1R1[i-1,...,1])*1e3
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
    l_C0_clip = np.all(np.vstack([cl_C1 <= C1R1_aux[:,0], C1R1_aux[:,0] <= cu_C1]), axis=0)
    l_R0_clip = np.all(np.vstack([cl_R1 <= C1R1_aux[:,1], C1R1_aux[:,1] <= cu_R1]), axis=0)
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
    
    
    
    ### Step 3: Make a proposal by generating samples from the hirachical 
    # model
    
    # sample from proposal for x at time t
    w_x = w((N,)).flatten()
    Sigma_X[i] = jax.vmap(
        functools.partial(model, I=ctrl_input[i-1], dt=dt))(
            x=Sigma_X[i-1,idx],
            C_1=(offset_C1+Sigma_C1R1[i-1,idx,0])*1e2,
            R_1=(offset_R1+Sigma_C1R1[i-1,idx,1])*1e3
        ) + w_x
    
    ## sample proposal for alpha and beta at time t
    # evaluate basis functions
    phi_0 = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i-1,idx])
    phi_1 = jax.vmap(functools.partial(basis_fcn))(Sigma_X[i])
    
    # calculate conditional predictive distribution for C1 and R1
    c_mean, c_col_scale, c_row_scale, df = jax.vmap(
        functools.partial(prior_mniw_CondPredictive, y1_var=y1_var)
        )(
            mean=GP_para_C1R1[0][idx],
            col_cov=GP_para_C1R1[1][idx],
            row_scale=GP_para_C1R1[2][idx],
            df=GP_para_C1R1[3][idx],
            y1=Sigma_C1R1[i-1,idx],
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
        GP_stats_C1R1[0][idx],
        GP_stats_C1R1[1][idx],
        GP_stats_C1R1[2][idx],
        GP_stats_C1R1[3][idx],
        Sigma_C1R1[i],
        phi_1
    ))
    
    # calculate new weights
    Sigma_Y[i] = jax.vmap(
        functools.partial(model.fy, I=ctrl_input[i], R_0=23e-3)
        )(x=Sigma_X[i])
    q_fy = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(Sigma_Y[i])
    q_C0_clip = np.all(np.vstack([cl_C1 <= Sigma_C1R1[i,:,0], Sigma_C1R1[i,:,0] <= cu_C1]), axis=0)
    q_R0_clip = np.all(np.vstack([cl_R1 <= Sigma_C1R1[i,:,1], Sigma_C1R1[i,:,1] <= cu_R1]), axis=0)
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
fig_Y, axes_Y = plot_Data(
    Particles=Sigma_Y[1:],
    weights=weights[1:],
    Reference=data["Voltage"].iloc[1:],
    time=data.index[1:]
)
axes_Y[0].set_ylabel(r"Voltage in $V$")
axes_Y[0].set_xlabel("Time")
apply_basic_formatting(fig_Y, width=8, font_size=8)
fig_Y.savefig("Battery_APF_Fsd.pdf", bbox_inches='tight')



# plot learned function
V = jnp.linspace(0, 2.2, 500)

Mean, Std, X_stats, X_weights, Time = generate_BFE_TimeSlices(
    N_slices=4, 
    X_in=V[...,None], 
    Sigma_X=Sigma_X, 
    Sigma_weights=weights,
    Mean=Mean_C1R1, 
    Col_Scale=Col_Cov_C1R1, 
    Row_Scale=Row_Scale_C1R1, 
    DF=df_C1R1, 
    basis_fcn=basis_fcn, 
    forget_factor=forget_factor
    )


fig_BFE, ax_BFE = plot_BFE_1D(
    V,
    np.array([offset_C1, offset_R1])[None,:,None] + Mean,
    Std,
    Time,
    X_stats, 
    X_weights
)
ax_BFE[0,0].set_ylabel(r"Capacity $C_1$ in $F \times 10^{2}$")
ax_BFE[0,1].set_ylabel(r"Resistance $R_1$ in $\Omega \times 10^{3}$")
for ax in ax_BFE[1]:
    ax.set_xlabel(r"Voltage in $V$")
apply_basic_formatting(fig_BFE, width=16, font_size=8)
fig_BFE.savefig("Battery_APF_Fsd_fcn.pdf", bbox_inches='tight')



plt.show()