import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from src.vehicle.Vehicle import forward_Bootstrap_PF, default_para



# load measurments
data = pd.read_csv('src/vehicle/measurment_3.csv', sep=';')
N = 10
    
U = data[['v_x']].to_numpy()
Y = data[['dpsi', 'v_y']].to_numpy()

# transition noise for steering angle
delta = data[['Steering_Angle']].to_numpy().flatten()
sigma_delta = np.var(delta[1:] - delta[:-1])

# Parameter
Q = np.diag([1e-3, 1e-3, sigma_delta])
R = np.diag([0.001/180*np.pi, (0.03/3.6)**2])

# generate initial particles
N = 5000
X_0 = data[['dpsi', 'v_y', 'Steering_Angle']].to_numpy()[0,:] + np.random.multivariate_normal([0, 0, 0], np.diag([0.001/180*np.pi, (0.03/3.6)**2, sigma_delta]), (N,))

# sample parameters
theta_mean = default_para[6:]
Sigma = np.diag([0.01, 0.01, *((theta_mean[2:]*0.1)**2)])

theta_sample = np.random.multivariate_normal(theta_mean, Sigma, (1,))

# start particle filter
X, w = forward_Bootstrap_PF(Y, U, X_0, theta_sample, Q, R)

# normalize weights
w = w / np.sum(w, axis=1, keepdims=True)

dpsi = np.sum(X[...,0]*w, axis=1)
v_y = np.sum(X[...,1]*w, axis=1)
delta = np.sum(X[...,2]*w, axis=1)



fig = make_subplots(3,1)

time = np.arange(data.index[0], data.index[-1]+default_para[0], default_para[0])
fig.add_trace(
    go.Scatter(
            x=time,
            y=dpsi,
            mode='lines',
            name='d_psi'
        ),
        row=1,
        col=1
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=v_y,
            mode='lines',
            name='v_y'
        ),
        row=2,
        col=1
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=delta,
            mode='lines',
            name='delta'
        ),
        row=3,
        col=1
    )

# plot Reference
fig.add_trace(
    go.Scatter(
            x=time,
            y=data[['dpsi']].to_numpy().flatten(),
            mode='lines',
            name='d_psi_ref'
        ),
        row=1,
        col=1
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=data[['v_y']].to_numpy().flatten(),
            mode='lines',
            name='v_y_ref'
        ),
        row=2,
        col=1
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=data[['Steering_Angle']].to_numpy().flatten(),
            mode='lines',
            name='delta_ref'
        ),
        row=3,
        col=1
    )

fig.show()