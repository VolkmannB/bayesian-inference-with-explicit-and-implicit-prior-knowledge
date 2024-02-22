import numpy as np
import jax
from tqdm import tqdm
import functools



from src.vehicle.Vehicle import features_MTF_front, features_MTF_rear, vehicle_RBF_ip, default_para, f_x_sim, f_y, fx_filter, fy_filter, f_alpha, mu_y, H_vehicle
from src.RGP import EnsambleGP, update_EIV_normal
from src.KalmanFilter import EnKF_update
from src.vehicle.VehiclePlotting import generate_Vehicle_Animation



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 40
t_end = 100.0
time = np.arange(0.0, t_end, default_para['dt'])
steps = len(time)


# model prior
prior = [
    np.zeros((vehicle_RBF_ip.shape[0],)),
    np.eye(vehicle_RBF_ip.shape[0])
]
noise = [
    np.zeros((vehicle_RBF_ip.shape[0],)),
    np.eye(vehicle_RBF_ip.shape[0])*1e-8
]

# model for the front tire
para_model_f = [
    prior[0],
    prior[1]
]

# model for the rear tire
para_model_r = [
    prior[0],
    prior[1]
]



# initial state
x0 = np.array([0.0, 0.0, 0.0, 0.0])
x_ = x0[0:2]
P0 = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
P_ = P0
x_part = np.zeros((N,3))

# noise
R = np.diag([0.01/180*np.pi, 1e-4])
Q = np.diag([5e-6, 5e-7, 1e-4, 1e-3])
w = lambda size=(): np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q, size)
e = lambda size=(): np.random.multivariate_normal(np.zeros((R.shape[0],)), R, size)



################################################################################
# Simulation

# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,2))

Sigma_X = np.zeros((steps,N,4))

W_f = np.zeros((steps, vehicle_RBF_ip.shape[0]))
CW_f = np.zeros((steps, vehicle_RBF_ip.shape[0], vehicle_RBF_ip.shape[0]))
W_r = np.zeros((steps, vehicle_RBF_ip.shape[0]))
CW_r = np.zeros((steps, vehicle_RBF_ip.shape[0], vehicle_RBF_ip.shape[0]))

# input
u = np.zeros((steps,2))
u[:,0] = 10/180*np.pi * np.sin(2*np.pi*time/5) * 0.5*(np.tanh(0.2*(time-15))-np.tanh(0.2*(time-75)))
u[:,1] = 11.0

# initial values
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
W_f[0,:] = para_model_f[0]
CW_f[0,:] = para_model_f[1]
W_r[0,:] = para_model_r[0]
CW_r[0,:] = para_model_r[1]

# initial training


# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    X[i] = f_x_sim(X[i-1], u[i-1], **default_para)
    Y[i] = f_y(X[i], u[i], **default_para) + e()
    
    
    
    ####### Filtering
    
    # time update
    Sigma_X[i,...] = jax.vmap(functools.partial(fx_filter, u=u[i-1], **default_para))(
        x = Sigma_X[i-1,...],
        theta_f=para_model_f[0] + (np.linalg.cholesky(para_model_f[1]) @ np.random.randn(vehicle_RBF_ip.shape[0],N)).T, 
        theta_r=para_model_r[0] + (np.linalg.cholesky(para_model_r[1]) @ np.random.randn(vehicle_RBF_ip.shape[0],N)).T
    ) + w((N,))
    
    # measurment update
    sigma_y = jax.vmap(functools.partial(fy_filter, u=u[i], **default_para))(
        x = Sigma_X[i,...]
    )
    Sigma_X[i,...] = EnKF_update(Sigma_X[i,...], sigma_y, np.concatenate([Y[i], [0]]), np.diag(np.concatenate([np.diag(R), [0.01]])))
    
    # update model front
    mu_x = np.mean(Sigma_X[i,...], axis=0)
    P_x = np.cov(Sigma_X[i,...].T)
    J_psi = jax.jacfwd(functools.partial(features_MTF_front, u=u[i], **default_para))(mu_x[:2])
    psi = features_MTF_front(x=mu_x[:2], u=u[i], **default_para)
    para_model_f = update_EIV_normal(
        psi,
        mu_x[2],
        P_x[2,2] - 2*para_model_f[0]@J_psi@P_x[:2,2] + para_model_f[0]@J_psi@P_x[:2,:2]@J_psi.T@para_model_f[0].T,
        *para_model_f
    )
    
    # update model back
    J_psi = jax.jacfwd(functools.partial(features_MTF_rear, u=u[i], **default_para))(mu_x[:2])
    psi = features_MTF_rear(x=mu_x[:2], u=u[i], **default_para)
    para_model_r = update_EIV_normal(
        psi,
        mu_x[3],
        P_x[3,3] - 2*psi@J_psi@P_x[:2,3] + psi@J_psi@P_x[:2,:2]@J_psi.T@psi.T,
        *para_model_r
    )
    
    para_model_f[1] += noise[1]
    para_model_r[1] += noise[1]
    
    # logging
    W_f[i,:] = para_model_f[0]
    CW_f[i,:] = para_model_f[1]
    W_r[i,:] = para_model_r[0]
    CW_r[i,:] = para_model_r[1]
    
    

fig = generate_Vehicle_Animation(X, Y, u, Sigma_X, W_f, CW_f, W_r, CW_r, time, default_para, 200., 30., 30)
fig.show()