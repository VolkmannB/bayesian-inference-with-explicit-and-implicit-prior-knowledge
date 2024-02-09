import numpy as np
import jax
from tqdm import tqdm



from src.vehicle.Vehicle import H_vehicle, vehicle_RBF_ip, default_para, f_x, f_y, fx_filter, fy_filter
from src.RGP import EnsambleGP, sq_dist
from src.KalmanFilter import EnsambleKalmanFilter



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 150
t_end = 100.0
dt = 0.01
time = np.arange(0.0,t_end,dt)
steps = len(time)

# measurment model for simulation
f_y_sim = lambda x, u: f_y(x, u, *default_para[1:])


# model for the front tire
mu_f_model = EnsambleGP(
    basis_function=H_vehicle,
    n_basis=len(vehicle_RBF_ip),
    w0=np.zeros(vehicle_RBF_ip.shape[0]),
    cov0=H_vehicle(vehicle_RBF_ip).T @ H_vehicle(vehicle_RBF_ip)*10,
    N=N,
    error_cov=0.001
)
mu_f_model.T = H_vehicle(vehicle_RBF_ip).T @ H_vehicle(vehicle_RBF_ip)

# model for the rear tire
mu_r_model = EnsambleGP(
    basis_function=H_vehicle,
    n_basis=len(vehicle_RBF_ip),
    w0=np.zeros(vehicle_RBF_ip.shape[0]),
    cov0=H_vehicle(vehicle_RBF_ip).T @ H_vehicle(vehicle_RBF_ip)*10,
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
meas_noise_var = np.diag([0.01/180*np.pi, 1e-3])
process_var = np.diag([5e-6, 5e-7, 1e-3, 1e-3])

fx_EnKF = lambda x, u, theta_f, theta_r: fx_filter(x, u, theta_f, theta_r, dt, *default_para[1:7])
fy_EnKF = lambda x, u: fy_filter(x, u, *default_para[1:7])

EnKF = EnsambleKalmanFilter(
    N,
    x0,
    P0,
    jax.vmap(fx_EnKF),
    jax.vmap(fy_EnKF),
    process_var,
    meas_noise_var
)


################################################################################
# Simulation

# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,2))
mu_yf = np.zeros((steps,))
mu_yr = np.zeros((steps,))

F_pred = np.zeros((steps,))
PF_pred = np.zeros((steps,))
Sigma_X = np.zeros((steps,N,2))

W = np.zeros((steps, vehicle_RBF_ip.shape[0]))
CW = np.zeros((steps, vehicle_RBF_ip.shape[0], vehicle_RBF_ip.shape[0]))

# input
u = np.zeros((steps,2))
u[:,1] = 10

# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    X[i] = f_x(X[i-1], u[i-1], *default_para)
    Y[i] = f_y_sim(X[i], u[i-1]) + np.random.multivariate_normal(np.zeros((2,)), meas_noise_var)
    
    
    
    ####### Filtering
    
    # time update
    EnKF.predict(u=np.repeat(u[i-1], N), theta_f=mu_f_model.W, theta_r=mu_r_model.W)
    
    # measurment update
    EnKF.update(Y[i], u=np.repeat(u[i-1], N))