import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.SingleMassOscillator import F_spring, F_damper, H, fx_KF, dx_KF, ip, f_x_sim, N_ip
from src.RGP import sq_dist, update_BMNIW_prior, sample_BMNIW_prior
from src.KalmanFilter import systematic_SISR, squared_error
from src.Plotting import generate_Animation



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
v = np.ones((N_ip**2,))
Lambda_0 = np.diag([5e-7, 5e-8])
GP_model = [
    np.zeros((N,2,2)),# Phi
    np.zeros((N,2,N_ip**2)), #Psi
    np.zeros((N,N_ip**2,N_ip**2)), #Sigma
    np.ones((N,))*2, # nu
]


# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
key = jax.random.key(np.random.randint(100, 1000))


# noise
R = np.array([[1e-2]])
Q = np.diag([5e-6, 5e-7])
w = lambda n=1: np.random.multivariate_normal(np.zeros((3,)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), R, n)



################################################################################
# Simulation


# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,))
F_sd = np.zeros((steps,))

F_pred = np.zeros((steps,)) # Filter
PF_pred = np.zeros((steps,))
Sigma_X = np.zeros((steps,N,2))

W = np.zeros((steps, N_ip**2)) # GP
CW = np.zeros((steps, N_ip**2, N_ip**2))

# input
F = np.zeros((steps,)) 
F[int(t_end/(5*dt)):] = -9.81*m
F[int(2*t_end/(5*dt)):] = -9.81*m*2
F[int(3*t_end/(5*dt)):] = -9.81*m
F[int(4*t_end/(5*dt)):] = 0

# set initial values
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
X[0,...] = x0
weights = np.ones((N,))/N



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
    key, *keys = jax.random.split(key, N+1)
    psi = jax.vmap(H)(Sigma_X[i-1])
    Sigma_X[i] = jax.vmap(functools.partial(sample_BMNIW_prior, v=v, Lambda_0=Lambda_0))(
        Phi = GP_model[0],
        Psi = GP_model[1],
        Sigma = GP_model[2],
        nu = GP_model[3],
        psi = psi,
        key = jnp.asarray(keys)
    )
    
    # measurment update
    sigma_y = Sigma_X[i,:,0,None]
    q = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(sigma_y)
    weights = weights * q
    weights = weights/np.sum(weights)
    if np.any(np.isnan(weights)):
        raise ValueError('PF divergence')
    
    # resampling
    u = np.random.rand()
    idx = systematic_SISR(u, weights)
    
    # copy statistics
    psi = psi[idx,...]
    Sigma_X[i,...] = Sigma_X[i,idx,...]
    GP_model[0] = GP_model[0][idx,...]
    GP_model[1] = GP_model[1][idx,...]
    GP_model[2] = GP_model[2][idx,...]
    GP_model[3] = GP_model[3][idx,...]
    weights = np.ones((N,))/N
    
    # update GP parameters
    GP_model = jax.vmap(update_BMNIW_prior)(
        *GP_model,
        Sigma_X[i],
        psi
    )
    
    # apply forgetting operator for t+1
    GP_model[0] *= 0.99
    GP_model[1] *= 0.99
    GP_model[2] *= 0.99
    GP_model[3] *= 0.99
    
    # logging
    # f = spring_damper_model.ensample_predict(Sigma_X[i,:,:2])
    # F_pred[i] = np.mean(f)
    # PF_pred[i] = np.var(f)
    # W[i,...] = spring_damper_model.W.mean(axis=0)
    # CW[i,...] = np.cov(spring_damper_model.W.T)
    
    

################################################################################
# Plots

fig = generate_Animation(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, W, CW, time, model_para, 200., 30., 30)

print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-F_pred)))

fig.show()