import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.SingleMassOscillator import F_spring, F_damper, f_x_sim, fx_KF, N_ip, ip, H
from src.RGP import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics, prior_mniw_sampleLikelihood
from src.KalmanFilter import systematic_SISR, squared_error
from src.Plotting import generate_Animation
from src.Algorithms import algorithm_PF_GibbsSampling_GP



################################################################################
# Model

rng = np.random.default_rng()

# sim para
N = 500
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
# parameters of the prior
phi = jax.vmap(H)(ip)
GP_model_prior_eta = list(prior_mniw_2naturalPara(
    np.zeros((1, N_ip)),
    phi@phi.T*40,
    np.eye(1),
    0
))

# parameters for the sufficient statistics
GP_model_stats = [
    np.zeros((N, N_ip, 1)),
    np.zeros((N, N_ip, N_ip)),
    np.zeros((N, 1, 1)),
    np.zeros((N,))
]

# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
X0 = np.random.multivariate_normal(x0, P0, (N,))
key = jax.random.key(np.random.randint(100, 1000))

# noise
R = np.array([[1e-3]])
Q = np.diag([5e-6, 5e-7])
w = lambda n=1: np.random.multivariate_normal(np.zeros((2,)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((1,)), R, n)

# function of the state space model
ssm_fcn = lambda x, ctrl_input, xi: fx_KF(x=x, F=ctrl_input, F_sd=jnp.atleast_1d(xi)[0], **model_para)

# measurment function
measurment_fcn = lambda x, ctrl_input: x[0]

# function of the likelihood
likelihood_fcn = lambda x, y: squared_error(jnp.atleast_1d(x), y, cov=R)

# basis function
basis_fcn = lambda x, ctrl_input: H(x)

# input signal
# F = np.ones((steps,)) * -9.81*m + np.sin(2*np.pi*np.arange(0,t_end,dt)/10) * 9.81
F = np.zeros((steps,)) 
F[int(t_end/(5*dt)):] = -9.81*m
F[int(2*t_end/(5*dt)):] = -9.81*m*2
F[int(3*t_end/(5*dt)):] = -9.81*m
F[int(4*t_end/(5*dt)):] = 0



################################################################################
# Simulation


# time series for plot
X = np.zeros((steps,2)) # sim
Y = np.zeros((steps,))
F_sd = np.zeros((steps,))

CW = np.zeros((steps, N_ip, N_ip))

# set initial values
phi = jax.vmap(H)(X0)
GP_model_stats = list(jax.vmap(prior_mniw_updateStatistics)( # initial value for GP
    *GP_model_stats,
    np.zeros((N,1)),
    phi
))
X[0,...] = x0
weights = np.ones((steps,N))/N



# simulation loop
for i in tqdm(range(1,steps), desc="Simulate System"):
    
    ####### Model simulation
    
    # update system state
    X[i] = f_x_sim(X[i-1], F[i-1], **model_para)
    F_sd[i] = F_spring(X[i,0], **model_para) + F_damper(X[i,1], **model_para)
    
    # generate measurment
    Y[i] = X[i,0] + e()[0,0]


Sigma_X, Sigma_F, W, weights, time = algorithm_PF_GibbsSampling_GP(
    N_basis_fcn=N_ip,
    N_xi=1,
    t_end=t_end,
    dt=dt,
    ctrl_input=F,
    Y=Y,
    X0=X0,
    GP_prior=[
        GP_model_prior_eta[0]+GP_model_stats[0][0],
        GP_model_prior_eta[1]+GP_model_stats[1][0],
        GP_model_prior_eta[2]+GP_model_stats[2][0],
        GP_model_prior_eta[3]+GP_model_stats[3][0]],
    basis_fcn=basis_fcn,
    ssm_fcn=ssm_fcn,
    measurment_fcn=measurment_fcn,
    likelihood_fcn=likelihood_fcn,
    noise_fcn=w
)
Sigma_F = Sigma_F.squeeze()
W = W.squeeze()
    
    

################################################################################
# Plots

fig = generate_Animation(X, Y, F_sd, Sigma_X, Sigma_F, weights, H, W, CW, time, model_para, 200., 30., 30)

# print('RMSE for spring-damper force is {0}'.format(np.std(F_sd-Sigma_F)))

fig.show()