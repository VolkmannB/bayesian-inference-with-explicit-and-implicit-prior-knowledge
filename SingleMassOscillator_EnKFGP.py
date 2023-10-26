import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import scipy.linalg



from src.SingleMassOscillator import SingleMassOscillator, f_model, F_spring, F_damper
from src.RGP import ApproximateGP, GaussianRBF
from src.sampling import condition_gaussian
from src.KalmanFilter import EnsambleKalmanFilter
from src.Plotting import generate_Animation, generate_Plot



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
model_para = {'m':m, 'c1':c1, 'c2':c2, 'd1':d1, 'd2':d2}

SMO = SingleMassOscillator(
    m=m,
    c1=c1,
    c2=c2,
    d1=d1,
    d2=d2,
    R=np.array([[1e-2]]))


# GP definition
x_points = np.arange(-5., 5.1, 1.)
dx_points = np.arange(-5., 5.1, 1.)
ip = np.dstack(np.meshgrid(x_points, dx_points, indexing='xy'))
ip = ip.reshape(ip.shape[0]*ip.shape[1], 2)
H = GaussianRBF(
    centers=ip,
    lengthscale=np.array([1.])
)


# EnKF
meas_noise_var = np.array([[1e-2]])
process_var = scipy.linalg.block_diag(np.diag([5e-6, 5e-7]), np.eye(ip.shape[0])*1e-12)
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
Y = np.zeros((steps,))
F_sd = np.zeros((steps,))

F_pred = np.zeros((steps,))
PF_pred = np.zeros((steps,))
Sigma_X = np.zeros((steps,N,2))
W = np.zeros((steps,ip.shape[0]))
CW = np.zeros((steps,ip.shape[0],ip.shape[0]))

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
    F_sd[i] = F_spring(x[0], c1, c2) + F_damper(x[1], d1, d2)
    
    # generate measurment
    Y[i] = y = SMO.measurent()[0,0]
    
    
    ####### Filtering
    
    EnKF.predict(F=F[i])
    EnKF.update(y)
    
    
    ####### Logging
    Sigma_X[i,...] = EnKF._sigma_x[:,:2]
    W[i,...] = EnKF.x[2:]
    CW[i,...] = EnKF.P[2:,2:]
    Hx = H(EnKF._sigma_x[:,:2])
    F_pred[i] = np.mean(Hx@ W[i,...])
    PF_pred[i] = np.var(Hx@ W[i,...])



################################################################################
# Plots

# generate_Animation(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, W, CW, time, model_para, 200., 30., 60)
generate_Plot(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, W[-1,...], CW[-1,...], time, model_para, 200., 30.)

plt.show()