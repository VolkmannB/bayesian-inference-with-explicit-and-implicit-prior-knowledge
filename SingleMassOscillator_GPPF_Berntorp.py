import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.SingleMassOscillator import F_spring, F_damper, f_x_sim, fx_KF, N_ip, H
from src.RGP import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics, prior_mniw_sampleLikelihood, gaussian_RBF
from src.KalmanFilter import systematic_SISR, squared_error
from src.Plotting import generate_Animation



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 100
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
GP_model_prior_eta = list(prior_mniw_2naturalPara(
    np.zeros((1, N_ip)),
    np.eye(N_ip)*20,
    np.eye(1)*1e-4,
    0
))
# GP_model_prior_eta[0] = np.repeat(GP_model_prior_eta[0][None,...], N, axis=0)
# GP_model_prior_eta[1] = np.repeat(GP_model_prior_eta[1][None,...], N, axis=0)
# GP_model_prior_eta[2] = np.repeat(GP_model_prior_eta[2][None,...], N, axis=0)
# GP_model_prior_eta[3] = np.repeat(GP_model_prior_eta[3][None,...], N, axis=0)

GP_model_stats = [
    np.zeros((N, N_ip, 1)),
    np.zeros((N, N_ip, N_ip)),
    np.zeros((N, 1, 1)),
    np.zeros((N,))
]


# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
key = jax.random.key(np.random.randint(100, 1000))


# noise
R = np.array([[1e-2]])
Q = np.diag([5e-6, 5e-7])
w = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), R, n)



################################################################################
# Simulation


# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,))
F_sd = np.zeros((steps,))

Sigma_X = np.zeros((steps,N,2))
Sigma_F = np.zeros((steps,N))

W = np.zeros((steps, N_ip)) # GP
CW = np.zeros((steps, N_ip, N_ip))

# input
# F = np.ones((steps,)) * -9.81*m + np.sin(2*np.pi*np.arange(0,t_end,dt)/10) * 9.81
F = np.zeros((steps,)) 
F[int(t_end/(5*dt)):] = -9.81*m
F[int(2*t_end/(5*dt)):] = -9.81*m*2
F[int(3*t_end/(5*dt)):] = -9.81*m
F[int(4*t_end/(5*dt)):] = 0

# set initial values
Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N,))
Sigma_F[0,...] = np.sqrt(1e-4) * np.random.randn(N)
X[0,...] = x0
weights = np.ones((steps,N))/N



# simulation loop
for i in tqdm(range(1,steps), desc="Running simulation"):
    
    ####### Model simulation
    
    # update system state
    X[i] = f_x_sim(X[i-1], F[i-1], **model_para)
    F_sd[i] = F_spring(X[i,0], **model_para) + F_damper(X[i,1], **model_para)
    
    # generate measurment
    Y[i] = X[i,0] + e()[0,0]
    
    
    ####### Filtering
    
    ## time update
    
    # apply forgetting operator for t+1
    GP_model_stats[0] *= 0.999
    GP_model_stats[1] *= 0.999
    GP_model_stats[2] *= 0.999
    GP_model_stats[3] *= 0.999
    
    # create auxiliary variable
    x_aux = jax.vmap(functools.partial(fx_KF, F=F[i-1], **model_para))(x=Sigma_X[i-1,...], F_sd=Sigma_F[i-1,...])
    
    # calculate first stage weights
    l = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(x_aux[:,0,None])
    p = weights[i-1] * l
    p = p/np.sum(p)
    
    # draw new indices
    u = np.random.rand()
    idx = systematic_SISR(u, p)
    
    # copy statistics
    GP_model_stats[0] = GP_model_stats[0][idx,...]
    GP_model_stats[1] = GP_model_stats[1][idx,...]
    GP_model_stats[2] = GP_model_stats[2][idx,...]
    GP_model_stats[3] = GP_model_stats[3][idx,...]
    
    # sample from proposal for x
    w_x = w((N,))
    Sigma_X[i] = jax.vmap(functools.partial(fx_KF, F=F[i-1], **model_para))(x=Sigma_X[i-1,idx,:], F_sd=Sigma_F[i-1,idx])
    
    # calculate parameters of posterior
    GP_posterior = list(jax.vmap(prior_mniw_2naturalPara_inv)(
        GP_model_prior_eta[0] + GP_model_stats[0],
        GP_model_prior_eta[1] + GP_model_stats[1],
        GP_model_prior_eta[2] + GP_model_stats[2],
        GP_model_prior_eta[3] + GP_model_stats[3]
    ))
    
    # sample from proposal for F
    phi = jax.vmap(H)(Sigma_X[i])
    key, *keys = jax.random.split(key, N+1)
    Sigma_F[i] = jax.vmap(prior_mniw_sampleLikelihood)(
        key=jnp.asarray(keys),
        M=GP_posterior[0],
        V=GP_posterior[1],
        Psi=GP_posterior[2],
        nu=GP_posterior[3],
        phi=phi
    ).flatten()
    
    # apply sampled noise on x
    Sigma_X[i] = Sigma_X[i] + w_x
    # Sigma_F[i] = F_spring(Sigma_X[i,:,0], **model_para) + F_damper(Sigma_X[i,:,1], **model_para)
    
    # update GP parameters
    GP_model_stats = list(jax.vmap(prior_mniw_updateStatistics)(
        *GP_model_stats,
        Sigma_F[i],
        phi
    ))
    
    # measurment update
    sigma_y = Sigma_X[i,:,0,None]
    q = jax.vmap(functools.partial(squared_error, y=Y[i], cov=R))(sigma_y)
    weights[i] = q / p[idx]
    weights[i] = weights[i]/np.sum(weights[i])
    
    # logging
    W[i,...] = np.sum(weights[i,:,None,None] * GP_posterior[0], axis=0).flatten()
    
    #abort
    if np.any(np.isnan(weights[i])):
        break
    
    

################################################################################
# Plots

fig = generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, H, W, CW, time, model_para, 200., 30., 30)

# print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-Sigma_F)))

fig.show()