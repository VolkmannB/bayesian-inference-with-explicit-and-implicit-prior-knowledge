import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import scipy.linalg



from src.SingleMassOscillator import SingleMassOscillator, f_model, F_spring, F_damper
from src.RGP import ApproximateGP, GaussianRBF
from src.sampling import condition_gaussian
from src.KalmanFilter import EnsambleKalmanFilter



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 600
t_end = 60.0
dt = 0.01
time = np.arange(0.0,t_end,dt)
steps = len(time)

# model para
m=2.0
c1=10.0
c2=2.0
d1=0.7
d2=0.4
    

SMO = SingleMassOscillator(
    m=m,
    c1=c1,
    c2=c2,
    d1=d1,
    d2=d2,
    R=np.array([[1e-2]]))


x_points = np.arange(-5., 5.1, 1.)
dx_points = np.arange(-5., 5.1, 1.)
ip = np.dstack(np.meshgrid(x_points, dx_points, indexing='xy'))
ip = ip.reshape(ip.shape[0]*ip.shape[1], 2)
H = GaussianRBF(
    centers=ip,
    lengthscale=np.array([1.])
)
spring_damper_model = ApproximateGP(
    basis_function=H,
    w0=np.zeros(ip.shape[0]),
    cov0=np.eye(ip.shape[0])*10**2,
    error_cov=0.001
)


# initial state
x0 = np.array([0.0, 0.0, 0.0])
x_ = x0
P0 = np.diag([1e-4, 1e-4, 1e-4])
P_ = P0
x_part = np.zeros((N,3))
spring_damper_model.update(np.array([[0.0, 0.0]]), np.array([[0.0]]), np.array([1e-4]))



# noise
# meas_noise_var = np.array([[1e-2]])
# process_var = np.diag([5e-6, 5e-7, 2e-2])

# fx_KF = lambda x, F: np.concatenate((f_model(x[:,0,None], x[:,1,None], F, x[:,2,None], m=m, dt=dt), x[:,2,None]), axis = 1)

meas_noise_var = np.array([[1e-2]])
process_var = scipy.linalg.block_diag(np.diag([5e-6, 5e-7]), np.eye(ip.shape[0])*1e-6)
x0 = np.zeros((ip.shape[0]+2,))
P0 = scipy.linalg.block_diag(np.diag([5e-6, 5e-7]), H(ip).T@H(ip)*10)
fx_KF = lambda x, F: np.concatenate((f_model(x[:,0,None], x[:,1,None], F, np.sum(H(x[:,:2])*x[:,2:], axis=1, keepdims=True), m=m, dt=dt), x[:,2:]), axis = 1)

EnKF = EnsambleKalmanFilter(
    N,
    x0,
    P0,
    fx_KF,
    lambda x: x[:,0,None],
    process_var,
    meas_noise_var
)



################################################################################
# Simulation


# time series for plot
X = np.zeros((steps,3)) # sim
X_pred = np.zeros((steps,2)) # estimate
P_pred = np.zeros((steps,2))
Y = np.zeros((steps,))

# input
F = np.zeros((steps,))
F[int(t_end/(3*dt)):] = -9.81*m
F[int(2*t_end/(3*dt)):] = -9.81*m*2



# simulation loop
for i in tqdm(range(0,steps), desc="Running simulation"):
    
    ####### Model simulation
    
    # update system state
    SMO.update(F[i], dt)
    X[i,:2] = x = SMO.x
    X[i,2] = F_spring(x[0], c1, c2) + F_damper(x[1], d1, d2)
    
    # generate measurment
    Y[i] = y = SMO.measurent()[0,0]
    
    
    ####### Filtering
    
    EnKF.predict(F=F[i])
    EnKF.update(y)
    
    # save estimate in GP
    # spring_damper_model.update(EnKF._sigma_x[:,:2], EnKF._sigma_x[:,2], np.ones(EnKF._sigma_x.shape[0])*EnKF.P[2,2].flatten())
    
    
    
    X_pred[i,:] = EnKF.x[:2]
    P_pred[i,:] = np.diag(EnKF.P[:2,:2])

spring_damper_model = ApproximateGP(
    basis_function=H,
    w0=EnKF.x[2:],
    cov0=EnKF.P[2:,2:],
    error_cov=0.001
)

################################################################################
# Plots

fig1, ax1 = plt.subplots(2,2)

x_plt = np.arange(-5., 5., 0.1)
dx_plt = np.arange(-5., 5., 0.1)

# create point grid
grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')

# calculate ground truth for spring damper force
grid_F = F_spring(grid_x, c1, c2) + F_damper(grid_y, d1, d2)

# calculate force from GP mapping
points = np.dstack([grid_x, grid_y])
# points = np.split(points.reshape((points.shape[0]*points.shape[1],2)), 100)
F_sd, pF_sd = spring_damper_model.predict(points[...,np.newaxis,:])
F_sd = np.squeeze(F_sd)
pF_sd = np.squeeze(pF_sd)
# F_sd = F_sd.reshape((grid_x.shape[0], grid_x.shape[1]))
# pF_sd = pF_sd.reshape((grid_x.shape[0], grid_x.shape[1]))

# error
e = np.abs(grid_F - F_sd)

# aplha from confidence
a = np.abs(1-np.sqrt(pF_sd)/10)

ax1[0,0].plot(ip[:,0], ip[:,1], marker='.', color='k', linestyle='none')
c = ax1[0,0].pcolormesh(grid_x, grid_y, e, label="Ground Truth", vmin=0, vmax=50, alpha=a)
ax1[0,0].plot(X[:,0], X[:,1], label="State Trajectory", color='r')
# ax1[0,0].plot(X_pred[:,0], X_pred[:,1], label="State Trajectory", color='r', linestyle='--')
ax1[0,0].set_xlabel("x")
ax1[0,0].set_ylabel("dx")
ax1[0,0].set_xlim(-5., 5.)
ax1[0,0].set_ylim(-5., 5.)
fig1.colorbar(c, ax=ax1[0,0])

# force
ax1[0,1].plot(time, X[:,2], label="F")
m_F, P_F = spring_damper_model.predict(X_pred[:,np.newaxis,:])
m_F = np.squeeze(m_F)
P_F = np.squeeze(P_F)
ax1[0,1].plot(time, m_F, label="F_pred")
lower = m_F - np.sqrt(P_F)
upper = m_F + np.sqrt(P_F)
ax1[0,1].fill_between(time, lower, upper, color='b', alpha=0.2)
ax1[0,1].set_xlabel("time in (s)")
ax1[0,1].set_ylabel("F in (N)")

# x
ax1[1,0].plot(time, Y, 'r.', label="Measurements")
lower = X_pred[:,0] - np.sqrt(P_pred[:,0])
upper = X_pred[:,0] + np.sqrt(P_pred[:,0])
ax1[1,0].fill_between(time, lower, upper, color='b', alpha=0.2)
ax1[1,0].plot(time, X[:,0], label="x")
ax1[1,0].plot(time, X_pred[:,0], label="x_pred")
ax1[1,0].legend()
ax1[1,0].set_xlabel("time in (s)")
ax1[1,0].set_ylabel("x in (m)")

# dx
lower = X_pred[:,1] - np.sqrt(P_pred[:,1])
upper = X_pred[:,1] + np.sqrt(P_pred[:,1])
ax1[1,1].fill_between(time, lower, upper, color='b', alpha=0.2)
ax1[1,1].plot(time, X[:,1], label="dx")
ax1[1,1].plot(time, X_pred[:,1], label="dx_pred")
ax1[1,1].set_xlabel("time in (s)")
ax1[1,1].set_ylabel("dx in (m/s)")

plt.show()