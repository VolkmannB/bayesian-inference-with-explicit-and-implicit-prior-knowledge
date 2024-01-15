import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



from src.SingleMassOscillator import SingleMassOscillator, f_model, F_spring, F_damper
from src.RGP import GaussianRBF, EnsambleGP, sq_dist
from src.KalmanFilter import EnsambleKalmanFilter
from src.Plotting import generate_Animation



################################################################################
# Functions

def Mahalanobis_distance(x, mu, P):
    
    S = np.linalg.cholesky(P+np.eye(P.shape[0])*1e-6)
    x_ = x-mu
    
    z = np.linalg.solve(S, x_.T)
    return np.sqrt(np.sum(z**2, axis=0))



def calculate_LOF(X, k):
    
    # calculate euclidean distance
    dist = np.sqrt(sq_dist(X, X))
    
    # sort distance
    sorted_idx = np.argsort(dist, axis=1)
    
    # distance for the k-th nearest neighbor
    k_dist = dist[:,sorted_idx[:,k]]
    
    # size of k-neighborhood // account for multiple equal distances
    num_k_neighbors = np.sum(dist <= k_dist[...,None], axis=1)-1
    
    # reachability distance for all points
    reach_dist = np.maximum(k_dist[None,...], dist)
    
    # local reachabilities
    lr = [reach_dist[i,sorted_idx[i,1:num_k_neighbors[i]]] for i in range(X.shape[0])]
    
    # local reachability density
    lrd = [np.mean(lr[i]) for i in range(X.shape[0])]
    
    # LOF
    LOF = [np.mean(lrd[sorted_idx[i,1:num_k_neighbors[i]]])/lrd[i] for i in range(X.shape[0])]
    
    return np.array(LOF)



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
spring_damper_model.T = H(ip).T@H(ip)


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
    
    # time update
    EnKF.predict(F=F[i])
    spring_damper_model.W += np.random.randn(*spring_damper_model.W.shape) @ np.eye(spring_damper_model.W.shape[-1])*1e-8 
    
    # generate spring damper force
    EnKF._sigma_x[:,2] = spring_damper_model.ensample_predict(EnKF._sigma_x[:,:2])
    
    # measurment update
    EnKF.update(y)
    
    # save estimate in GP
    spring_damper_model.ensample_update(
        EnKF._sigma_x[:,:2], 
        EnKF._sigma_x[:,2], 
        np.var(EnKF._sigma_x[:,2])
        )
    
    # resampling
    is_outlier = get_outlier(EnKF._sigma_x[:,:2], k=5, r=10)
    if np.any(is_outlier):
        temp_mean = np.mean(EnKF._sigma_x[~is_outlier,:2], axis=0)
        temp_cov = np.cov(EnKF._sigma_x[~is_outlier,:2].T)
        temp_cov += np.eye(temp_cov.shape[0])*1e-6
        temp_L = np.linalg.cholesky(temp_cov)
        EnKF._sigma_x[is_outlier,:2] = temp_mean + np.random.randn(np.sum(is_outlier),2) @ temp_L.T
        
    #     temp_mean = np.mean(spring_damper_model.W[~is_outlier,:], axis=0)
    #     temp_cov = np.cov(spring_damper_model.W[~is_outlier,:].T)*spring_damper_model.T
    #     temp_cov += np.eye(temp_cov.shape[0])*1e-6
    #     temp_L = np.linalg.cholesky(temp_cov)
    #     spring_damper_model.W[is_outlier,:] = temp_mean + np.random.randn(np.sum(is_outlier),temp_cov.shape[1]) @ temp_L.T
    
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

# fig, ax = plt.subplots(1,2)
# pos1 = ax[0].imshow(spring_damper_model.cov)
# fig.colorbar(pos1, ax=ax[0])

# pos2 = ax[1].imshow(H(ip).T@H(ip))
# fig.colorbar(pos2, ax=ax[1])

fig.show()