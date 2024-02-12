import numpy as np
from tqdm import tqdm
import jax
import functools



from src.SingleMassOscillator import F_spring, F_damper, H, fx_KF, dx_KF, ip, f_x_sim
from src.RGP import EnsambleGP, sq_dist
from src.KalmanFilter import EnKF_update
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
model_para = {'dt':dt, 'm':m, 'c1':c1, 'c2':c2, 'd1':d1, 'd2':d2}


# model of the spring damper system
spring_damper_model = EnsambleGP(
    basis_function=H,
    n_basis=len(ip),
    w0=np.zeros(ip.shape[0]),
    cov0=H(ip).T@H(ip)*10,
    N=N,
    error_cov=0.001
)
spring_damper_model.T = H(ip).T@H(ip)


# initial system state
x0 = np.array([0.0, 0.0, 0.0])
P0 = np.diag([1e-4, 1e-4, 1e-4])


# noise
R = np.array([[1e-2]])
Q = np.diag([5e-6, 5e-7, 1e-3])
w = lambda n=...: np.random.multivariate_normal(np.zeros((3,)), Q)
e = lambda n=...: np.random.multivariate_normal(np.zeros((1,)), R)



################################################################################
# Simulation


# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,))
F_sd = np.zeros((steps,))

F_pred = np.zeros((steps,)) # Filter
PF_pred = np.zeros((steps,))
Sigma_X = np.zeros((steps,N,3))

W = np.zeros((steps,ip.shape[0])) # GP
CW = np.zeros((steps,ip.shape[0],ip.shape[0]))

# input
F = np.zeros((steps,)) 
F[int(t_end/(5*dt)):] = -9.81*m
F[int(2*t_end/(5*dt)):] = -9.81*m*2
F[int(3*t_end/(5*dt)):] = -9.81*m
F[int(4*t_end/(5*dt)):] = 0

# set initial values
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
X[0,...] = x0[:2]

# initial training of model
spring_damper_model.ensample_update(
    Sigma_X[0,:,:2], 
    Sigma_X[0,:,2], 
    P0[2,2]
    )



# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    
    # update system state
    X[i] = f_x_sim(X[i-1], F[i-1], **model_para)
    F_sd[i] = F_spring(X[i,0], **model_para) + F_damper(X[i,1], **model_para)
    
    # generate measurment
    Y[i] = X[i,0] + e()[0]
    
    
    ####### Filtering
    
    # time update
    Sigma_X[i] = jax.vmap(functools.partial(fx_KF, F=F[i-1], **model_para))(
        x= Sigma_X[i-1],
        theta= spring_damper_model.W
    ) + w((N,))
    
    # measurment update
    sigma_y = Sigma_X[i,:,0] #+ e((N,))
    Sigma_X[i] = EnKF_update(Sigma_X[i], sigma_y[:,np.newaxis], Y[i], R)
    
    # save estimate in GP
    spring_damper_model.ensample_update(
        Sigma_X[i,:,:2], 
        Sigma_X[i,:,2], 
        np.var(Sigma_X[i,:,2])
        )
    
    # resampling
    is_outlier = get_outlier(Sigma_X[i], k=5, r=10)
    if np.any(is_outlier):
        temp_mean = np.mean(Sigma_X[i,~is_outlier,:], axis=0)
        temp_cov = np.cov(Sigma_X[i,~is_outlier,:].T)
        temp_cov += np.eye(temp_cov.shape[0])*1e-6
        temp_L = np.linalg.cholesky(temp_cov)
        Sigma_X[i,is_outlier,:] = temp_mean + np.random.randn(np.sum(is_outlier),temp_cov.shape[0]) @ temp_L.T
    
    # logging
    f = spring_damper_model.ensample_predict(Sigma_X[i,:,:2])
    F_pred[i] = np.mean(f)
    PF_pred[i] = np.var(f)
    W[i,...] = spring_damper_model.W.mean(axis=0)
    CW[i,...] = np.cov(spring_damper_model.W.T)
    
    

################################################################################
# Plots

fig = generate_Animation(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, W, CW, time, model_para, 200., 30., 30)

print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-F_pred)))

fig.show()