import numpy as np
from tqdm import tqdm
import jax



from src.SingleMassOscillator import SingleMassOscillator, F_spring, F_damper, H, fx_KF, dx_KF, ip
from src.RGP import EnsambleGP, sq_dist
from src.KalmanFilter import EnsambleKalmanFilter
from src.Plotting import generate_Animation



################################################################################
# Functions

def get_outlier(X, k, r=1.5):
    
    # calculate euclidean distance
    dist = np.sqrt(sq_dist(X, X))
    
    # sort distance
    sorted_dist = np.sort(dist, axis=1)
    
    # distance for the k-th nearest neighbor
    k_dist = sorted_dist[:,k]
    
    # size of k-neighborhood // account for multiple equal distances
    num_k_neighbors = np.sum(dist <= k_dist[...,None], axis=1)-1
    
    # median local distance
    if np.all(num_k_neighbors[0] == num_k_neighbors[1:]):
        mld = np.median(sorted_dist[:,1:num_k_neighbors[0]], axis=1)
    else:
        mld = np.array([np.median(sorted_dist[i,1:num_k_neighbors[i]]) for i in range(X.shape[0])])
    
    quant3, quant1 = np.percentile(mld, [75 ,25])
    iqr = quant3-quant1
    return np.logical_or(mld > quant3+ r * iqr, mld < quant1- r * iqr)



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 150
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


spring_damper_model = EnsambleGP(
    basis_function=H,
    n_basis=len(ip),
    w0=np.zeros(ip.shape[0]),
    cov0=H(ip).T@H(ip)*10,
    N=N,
    error_cov=0.001
)
spring_damper_model.T = H(ip).T@H(ip)


# initial state
x0 = np.array([0.0, 0.0, 0.0])
x_ = x0
P0 = np.diag([1e-4, 1e-4, 1e-4])
P_ = P0
x_part = np.zeros((N,3))



# noise
meas_noise_var = np.array([[1e-2]])
process_var = np.diag([5e-6, 5e-7, 1e-3])

fx_EnKF = lambda x, F, theta: fx_KF(dx_KF, x, m, F, theta, dt)

EnKF = EnsambleKalmanFilter(
    N,
    x0,
    P0,
    jax.jit(fx_EnKF),
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
    
    # time update
    EnKF.predict(F=F[i], theta=spring_damper_model.W)
    
    # measurment update
    EnKF.update(y)
    
    # save estimate in GP
    spring_damper_model.ensample_update(
        EnKF._sigma_x[:,:2], 
        EnKF._sigma_x[:,2], 
        np.var(EnKF._sigma_x[:,2])
        )
    
    # resampling
    is_outlier = get_outlier(EnKF._sigma_x, k=5, r=10)
    if np.any(is_outlier):
        temp_mean = np.mean(EnKF._sigma_x[~is_outlier,:], axis=0)
        temp_cov = np.cov(EnKF._sigma_x[~is_outlier,:].T)
        temp_cov += np.eye(temp_cov.shape[0])*1e-6
        temp_L = np.linalg.cholesky(temp_cov)
        EnKF._sigma_x = EnKF._sigma_x.at[is_outlier,:].set(temp_mean + np.random.randn(np.sum(is_outlier),temp_cov.shape[0]) @ temp_L.T)
    
    # logging
    f = spring_damper_model.ensample_predict(EnKF._sigma_x[:,:2])
    F_pred[i] = np.mean(f)
    PF_pred[i] = np.var(f)
    Sigma_X[i,...] = EnKF._sigma_x[:,:2]
    W[i,...] = spring_damper_model.W.mean(axis=0)
    CW[i,...] = np.cov(spring_damper_model.W.T)
    
    

################################################################################
# Plots

fig = generate_Animation(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, W, CW, time, model_para, 200., 30., 30)

print('RMSE for spring-damper force is {0}'.format(np.sqrt( ((F_sd-F_pred)**2).mean() )))

fig.show()