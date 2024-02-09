import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal
import scipy.io


from src.vehicle.Vehicle import forward_Bootstrap_PF, default_para



# load measurments
file_name = 'Measurment_3'
data = pd.read_csv(f'src/vehicle/{file_name}.csv', sep=';')
U = data[['v_x']].to_numpy()
Y = data[['dpsi', 'v_y', 'Inv_Path_Radius']].to_numpy()

# Noise Parameter
Q = np.diag([1e-3, 1e-3, 0.02])
R = np.diag([0.001/180*np.pi, (0.03/3.6)**2, 1e-3])

# sample parameters
theta_mean = default_para[6:]
Sigma = np.diag([0.01, 0.01, *((theta_mean[2:]*0.1)**2)])

# Number of Particles
N = 5000
M = 10

# Logging
Theta = np.zeros((M, theta_mean.shape[0]))
Delta = np.zeros((M, data.shape[0]))

### Initiate PMCMC
print('Initiating PMCMC')

# make initial proposal
Theta[0,:] = np.random.multivariate_normal(theta_mean, Sigma, (1,))

# generate initial particles
X_0 = data[['dpsi', 'v_y', 'Steering_Angle']].to_numpy()[0,:] + np.random.multivariate_normal([0, 0, 0], np.diag([0.001/180*np.pi, (0.03/3.6)**2, Q[2,2]]), (N,))

# Initial PF
X, log_w = forward_Bootstrap_PF(Y, U, X_0, Theta[0,:], Q, R)
    
# save smoothed steering angle
w = np.exp(log_w[-1,...]) / np.sum(np.exp(log_w[-1,...]))
Delta[0,:] = np.sum(X[...,2]*w, axis=1)

# likelihood of proposal
log_P_y_theta_old = np.sum(log_w)

### Start Iteration
for i in np.arange(1, M):
    print(f'\nStarting PMCMC iteration {i}')

    # propose a new parameter set
    theta_sample = np.random.multivariate_normal(Theta[i-1,:], 0.5**2*Sigma, (1,)).flatten()

    # generate initial particles
    X_0 = data[['dpsi', 'v_y', 'Steering_Angle']].to_numpy()[0,:] + np.random.multivariate_normal([0, 0, 0], np.diag([0.001/180*np.pi, (0.03/3.6)**2, Q[2,2]]), (N,))

    # start particle filter
    while True:
        try:
            X, log_w = forward_Bootstrap_PF(Y, U, X_0, theta_sample, Q, R)
            break
        except:
            print('Experienced particle depletion and starting over')

    # likelihood of proposal
    log_P_y_theta_new = np.sum(log_w)
    
    # likelihood from prior
    e_old = theta_mean - Theta[i-1,:]
    e_new = theta_mean - theta_sample
    log_P_theta_old = -0.5* e_old @ np.linalg.solve(Sigma, e_old)
    log_P_theta_new = -0.5* e_new @ np.linalg.solve(Sigma, e_new)
    
    # calculate acceptance probability
    alpha = np.minimum(np.log(1), -log_P_y_theta_old - log_P_theta_old + log_P_y_theta_new + log_P_theta_new)
    
    # accept or reject
    d = np.log(np.random.rand())
    if d < alpha:
        # save parameter proposal
        Theta[i,:] = theta_sample
        print('Particle accepted')
        
        # save smoothed steering angle
        w = np.exp(log_w[-1,...]) / np.sum(np.exp(log_w[-1,...]))
        Delta[i,:] = np.sum(X[...,2]*w, axis=1)
        
        # save old likelihood 
        log_P_y_theta_old = log_P_y_theta_new
        
    else:
        Theta[i,:] = Theta[i-1,:]
        Delta[i,:] = Delta[i-1,:]
        print('Particle rejected')
    


dpsi = np.sum(X[...,0]*w, axis=1)
v_y = np.sum(X[...,1]*w, axis=1)
delta = np.mean(Delta, axis=0)


# filter steering angle
sos4 = scipy.signal.butter(4, 0.07, output='sos')
delta = scipy.signal.sosfiltfilt(sos4, delta)

# save data
scipy.io.savemat(f'{file_name}.mat', {'Theta': Theta, 'Delta': Delta, 'delta_filt': delta})

# plot results
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
            name='d_psi_ref',
            line=dict(dash='dot')
        ),
        row=1,
        col=1
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=data[['v_y']].to_numpy().flatten(),
            mode='lines',
            name='v_y_ref',
            line=dict(dash='dot')
        ),
        row=2,
        col=1
    )

fig.add_trace(
    go.Scatter(
            x=time,
            y=data[['Steering_Angle']].to_numpy().flatten(),
            mode='lines',
            name='delta_ref',
            line=dict(dash='dot')
        ),
        row=3,
        col=1
    )

L = default_para[3] + default_para[4]
fig.add_trace(
    go.Scatter(
            x=time,
            y=np.arctan(data[['Inv_Path_Radius']].to_numpy().flatten()*L),
            mode='lines',
            name='delta_InvPath',
            line=dict(dash='dot')
        ),
        row=3,
        col=1
    )

fig.show()
fig.write_html('PMCMC.html')