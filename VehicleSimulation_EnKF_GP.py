import numpy as np
import jax
from tqdm import tqdm
import functools



from src.vehicle.Vehicle import H_vehicle, vehicle_RBF_ip, default_para, f_x_sim, f_y, fx_filter, fy_filter, f_alpha, mu_y
from src.RGP import EnsambleGP
from src.KalmanFilter import EnKF_update
from src.vehicle.VehiclePlotting import generate_Vehicle_Animation



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 100
t_end = 100.0
time = np.arange(0.0, t_end, default_para['dt'])
steps = len(time)



# model for the front tire
mu_f_model = EnsambleGP(
    basis_function=H_vehicle,
    n_basis=len(vehicle_RBF_ip),
    w0=np.zeros(vehicle_RBF_ip.shape[0]),
    cov0=H_vehicle(vehicle_RBF_ip).T @ H_vehicle(vehicle_RBF_ip),
    N=N,
    error_cov=0.001
)
mu_f_model.T = H_vehicle(vehicle_RBF_ip).T @ H_vehicle(vehicle_RBF_ip)

# model for the rear tire
mu_r_model = EnsambleGP(
    basis_function=H_vehicle,
    n_basis=len(vehicle_RBF_ip),
    w0=np.zeros(vehicle_RBF_ip.shape[0]),
    cov0=H_vehicle(vehicle_RBF_ip).T @ H_vehicle(vehicle_RBF_ip),
    N=N,
    error_cov=0.001
)
mu_r_model.T = H_vehicle(vehicle_RBF_ip).T @ H_vehicle(vehicle_RBF_ip)



# initial state
x0 = np.array([0.0, 0.0, 0.0, 0.0])
x_ = x0[0:2]
P0 = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
P_ = P0
x_part = np.zeros((N,3))

# noise
R = np.diag([0.01/180*np.pi, 1e-3])
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
u[:,0] = 10/180*np.pi * np.sin(2*np.pi*time/5) * 0.5*(np.tanh(0.2*(time-15))-np.tanh(0.2*(time-85)))
u[:,1] = 8.0

# initial values
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
W_f[0,:] = np.mean(mu_f_model.W, axis=0)
CW_f[0,:] = np.cov(mu_f_model.W.T)
W_r[0,:] = np.mean(mu_r_model.W, axis=0)
CW_r[0,:] = np.cov(mu_r_model.W.T)

# initial training
alpha_f, alpha_r = jax.vmap(functools.partial(f_alpha, u=u[0], **default_para))(x=Sigma_X[0,:,0:2])
mu_f_model.ensample_update(
    np.atleast_2d(alpha_f).T,
    Sigma_X[0,:,2],
    np.var(Sigma_X[0,:,2])
)
mu_r_model.ensample_update(
    np.atleast_2d(alpha_r).T,
    Sigma_X[0,:,3],
    np.var(Sigma_X[0,:,3])
)

# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    X[i] = f_x_sim(X[i-1], u[i-1], **default_para)
    Y[i] = f_y(X[i], u[i-1], **default_para) + e()
    
    
    
    ####### Filtering
    
    # time update
    Sigma_X[i,...] = jax.vmap(functools.partial(fx_filter, u=u[i-1], **default_para))(
        x = Sigma_X[i-1,...],
        theta_f = mu_f_model.W,
        theta_r = mu_r_model.W
    ) + w((N,))
    
    # measurment update
    sigma_y = jax.vmap(functools.partial(fy_filter, u=u[i-1], **default_para))(
        x = Sigma_X[i,...]
    )
    Sigma_X[i,...] = EnKF_update(Sigma_X[i,...], sigma_y, Y[i], R)
    
    # update models
    alpha_f, alpha_r = jax.vmap(functools.partial(f_alpha, u=u[i-1], **default_para))(x=Sigma_X[i,:,0:2])
    mu_f_model.ensample_update(
        np.atleast_2d(alpha_f).T,
        Sigma_X[i,:,2],
        np.var(Sigma_X[i,:,2])
    )
    mu_r_model.ensample_update(
        np.atleast_2d(alpha_r).T,
        Sigma_X[i,:,3],
        np.var(Sigma_X[i,:,3])
    )
    
    # logging
    W_f[i,:] = np.mean(mu_f_model.W, axis=0)
    CW_f[i,:] = np.cov(mu_f_model.W.T)
    W_r[i,:] = np.mean(mu_r_model.W, axis=0)
    CW_r[i,:] = np.cov(mu_r_model.W.T)
    
    

fig = generate_Vehicle_Animation(X, Y, u, Sigma_X, W_f, CW_f, W_r, CW_r, time, default_para, 200., 30., 30)
fig.show()