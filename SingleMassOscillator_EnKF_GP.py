import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import scipy.linalg



from src.SingleMassOscillator import SingleMassOscillator, f_model, F_spring, F_damper
from src.RGP import ApproximateGP, GaussianRBF, EnsambleGP
from src.sampling import condition_gaussian
from src.KalmanFilter import EnsambleKalmanFilter
from src.Plotting import generate_Animation, generate_Plot



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 300
t_end = 100.0
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
spring_damper_model = EnsambleGP(
    basis_function=H,
    w0=np.zeros(ip.shape[0]),
    cov0=H(ip).T@H(ip)*10,
    N=N,
    error_cov=0.001
)


# initial state
x0 = np.array([0.0, 0.0, 0.0])
x_ = x0
P0 = np.diag([1e-4, 1e-4, 1e-4])
P_ = P0
x_part = np.zeros((N,3))



# noise
meas_noise_var = np.array([[1e-2]])
process_var = np.diag([5e-6, 5e-7, 0.5])

fx_KF = lambda x, F: np.concatenate((f_model(x[:,0,None], x[:,1,None], F, x[:,2,None], m=m, dt=dt), x[:,2,None]), axis = 1)

EnKF = EnsambleKalmanFilter(
    N,
    x0,
    P0,
    fx_KF,
    lambda x: x[:,0,None],
    process_var,
    meas_noise_var
)

spring_damper_model.ensample_update(
    EnKF._sigma_x[:,:2], 
    EnKF._sigma_x[:,2], 
    P0[2,2]
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
F[int(t_end/(5*dt)):] = -9.81*m
F[int(2*t_end/(5*dt)):] = -9.81*m*2
F[int(3*t_end/(5*dt)):] = -9.81*m
F[int(4*t_end/(5*dt)):] = 0



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
    
    # generate spring damper force
    # EnKF._sigma_x[:,2] = spring_damper_model.ensample_predict(EnKF._sigma_x[:,:2])
    
    EnKF.predict(F=F[i])
    
    # generate spring damper force
    EnKF._sigma_x[:,2] = spring_damper_model.ensample_predict(EnKF._sigma_x[:,:2])
    
    EnKF.update(y)
    
    # save estimate in GP
    spring_damper_model.ensample_update(
        EnKF._sigma_x[:,:2], 
        EnKF._sigma_x[:,2], 
        np.var(EnKF._sigma_x[:,2])
        )
    
    
    # logging
    F_pred[i] = EnKF.x[2]
    PF_pred[i] = EnKF.P[2,2]
    Sigma_X[i,...] = EnKF._sigma_x[:,:2]
    W[i,...] = spring_damper_model.W.mean(axis=0)
    CW[i,...] = np.cov(spring_damper_model.W.T)
    
    

################################################################################
# Plots

# generate_Animation(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, W, CW, time, model_para, 200., 30., 30)
generate_Plot(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, spring_damper_model.mean, spring_damper_model.cov, time, model_para, 200., 30.)

plt.show()