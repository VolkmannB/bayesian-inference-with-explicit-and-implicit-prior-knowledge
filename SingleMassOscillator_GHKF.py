import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import scipy.linalg



from src.SingleMassOscillator import SingleMassOscillator, f_model, F_spring, F_damper
from src.RGP import ApproximateGP, GaussianRBF
from src.sampling import condition_gaussian
from src.KalmanFilter import GaussHermiteKalmanFilter
from src.Plotting import generate_Animation, generate_Plot



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 100
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
    cov0=H(ip).T@H(ip)*10,
    jitter_val=0.001
)


# initial state
x0 = np.array([0.0, 0.0, 0.0])
x_ = x0
P0 = np.diag([1e-4, 1e-4, 1e-4])
P_ = P0
x_part = np.zeros((N,3))
spring_damper_model.update(np.array([[0.0, 0.0]]), np.array([[0.0]]), np.array([1e-4]))



# noise
meas_noise_var = np.array([[1e-2]])
process_var = np.diag([5e-6, 5e-7, 0.5])

fx_KF = lambda x, F: np.concatenate((f_model(x[:,0,None], x[:,1,None], F, x[:,2,None], m=m, dt=dt), x[:,2,None]), axis = 1)

GHKF = GaussHermiteKalmanFilter(
    x0,
    P0,
    fx_KF,
    lambda x: x[:,0,None],
    process_var,
    meas_noise_var,
    11
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
    
    GHKF.predict(F=F[i])
    GHKF.update(y)
    
    # save estimate in GP
    sigma_x = GHKF._x + np.random.randn(N,3) @ np.linalg.cholesky(GHKF._P).T
    ind = np.array(np.floor(np.random.rand(10)*N), dtype=np.int64)
    spring_damper_model.update(sigma_x[ind,:2], sigma_x[ind,2], np.ones(len(ind))*GHKF.P[2,2].flatten())
    
    
    # logging
    F_pred[i] = GHKF.x[2]
    PF_pred[i] = GHKF.P[2,2]
    Sigma_X[i,...] = sigma_x[:,:2]
    
    

################################################################################
# Plots

generate_Plot(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, spring_damper_model._mean, spring_damper_model._cov, time, model_para, 200., 30.)

plt.show()