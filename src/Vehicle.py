import numpy as np
import jax
import jax.numpy as jnp
import functools
from tqdm import tqdm


from src.BayesianInferrence import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.BayesianInferrence import prior_mniw_2naturalPara_inv
from src.BayesianInferrence import prior_mniw_calcStatistics
from src.BayesianInferrence import prior_mniw_CondPredictive
from src.Filtering import systematic_SISR, log_likelihood_Normal, log_likelihood_Multivariate_t


#### This section defines the state space model

##### default arameters
m = 1720.0
I_zz = 1827.5
l_f = 1.16
l_r = 1.47
g = 9.81
mu_x = 0.9
mu = 0.9
B = 10.0
C = 1.9
E = 0.97


# tire load
def f_Fz(m, l_f, l_r, g):
    
    l = l_f + l_r
    mg = m*g
    F_zf = mg*l_r/l
    F_zr = mg*l_f/l
    
    return F_zf, F_zr



# friction MTF curve
@jax.jit
def mu_y(alpha, mu=mu, B=B, C=C, E=E):
    
    return mu * jnp.sin(C * jnp.arctan(B*(1-E)*jnp.tan(alpha) + E*jnp.arctan(B*jnp.tan(alpha))))



# side slip
@jax.jit
def f_alpha(x, u, l_f=l_f, l_r=l_r):

    vx_f = u[1]
    vy_f = x[1] + x[0]*l_f
    
    vx_r = u[1]
    vy_r = x[1] - x[0]*l_r
    
    return u[0]-jnp.arctan(vy_f/vx_f), -jnp.arctan(vy_r/vx_r)



# state dynamics
def dx(x, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x):
    
    F_zf, F_zr = f_Fz(m, l_f, l_r, g)
    
    dv_y = 1/m * (F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*mu_x*jnp.sin(u[0])) - u[1]*x[0]
    ddpsi = 1/I_zz * (l_f*F_zf*mu_yf*jnp.cos(u[0]) - l_r*F_zr*mu_yr + l_f*F_zf*mu_x*jnp.sin(u[0]))
    
    return jnp.stack([ddpsi, dv_y])



# time discrete state space model with Runge-Kutta-4
@jax.jit
def f_x(x, u, mu_yf, mu_yr, dt, m=m, I_zz=I_zz, l_f=l_f, l_r=l_r, g=g, mu_x=mu_x):
    
    k1 = dx(x, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k2 = dx(x + dt*k1/2.0, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k3 = dx(x + dt*k2/2.0, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k4 = dx(x + dt*k3, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    
    return x + dt/6.0*(k1+2*k2+2*k3+k4)



# measurment model
@jax.jit
def f_y(x, u, mu_yf, mu_yr, m=m, l_f=l_f, l_r=l_r, g=g, mu_x=mu_x, mu=mu, B=B, C=C, E=E):
    
    F_zf, F_zr = f_Fz(m, l_f, l_r, g)
    
    dv_y = 1/m * (F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*mu_x*jnp.sin(u[0])) - u[1]*x[0]
    
    return jnp.array([x[0], dv_y, x[1]])



#### This section defines relevant parameters for the simulation

# set seed for reproducability
np.random.seed(16723573)

# simulation parameters
N_particles = 200
forget_factor = 0.999
dt = 0.01
t_end = 100.0
time = np.arange(0.0, t_end, dt)
steps = len(time)

# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])

# noise
R = np.diag([0.01/180*np.pi, 1e-1, 1e-3])
Q = np.diag([1e-9, 1e-9])
R_y = 1e1
w = lambda n=1: np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q, n)
e = lambda n=1: np.random.multivariate_normal(np.zeros((R.shape[0],)), R, n)


# control input to the vehicle as [steering angle, longitudinal velocity]
ctrl_input = np.zeros((steps,2))
ctrl_input[:,0] = 10/180*np.pi * np.sin(2*np.pi*time/5) * 0.5*(np.tanh(0.2*(time-15))-np.tanh(0.2*(time-75)))
ctrl_input[:,1] = 11.0



#### This section defines the basis function expansion

# basis functions for front and rear tire
N_basis_fcn = 10
lengthscale = 2 * 20/180*jnp.pi / N_basis_fcn
basis_fcn, spectral_density = generate_Hilbert_BasisFunction(
    N_basis_fcn, 
    np.array([-30/180*jnp.pi, 30/180*jnp.pi]), 
    lengthscale, 
    50
    )

@jax.jit
def features_MTF_front(x, u, l_f=l_f):
    
    vx = u[1]
    vy = x[1] + x[0]*l_f
    alpha = u[0] - jnp.arctan(vy/vx)
    
    return basis_fcn(alpha)

@jax.jit
def features_MTF_rear(x, u, l_r=l_r):
    
    vx = u[1]
    vy = x[1] - x[0]*l_r
    alpha = -jnp.arctan(vy/vx)
    
    return basis_fcn(alpha)



# model prior front tire
GP_prior_f = list(prior_mniw_2naturalPara(
    np.zeros((1, N_basis_fcn)),
    np.diag(spectral_density),
    np.eye(1)*1e-1,
    0
))

# model prior rear tire
GP_prior_r = list(prior_mniw_2naturalPara(
    np.zeros((1, N_basis_fcn)),
    np.eye(N_basis_fcn),
    np.eye(1)*1e-1,
    0
))



#### This section defines a function for the simulation of the system

def Vehicle_simulation():

    # time series for plot
    X = np.zeros((steps,2)) # sim
    Y = np.zeros((steps,3))
    
    # initial value
    X[0,...] = x0
        
        
    # simulation loop
    for i in tqdm(range(1,steps), desc="Running simulation"):
        
        ####### Model simulation
        alpha_f, alpha_r = f_alpha(X[i-1], ctrl_input[i-1])
        mu_f = mu_y(alpha_f)
        mu_r = mu_y(alpha_r)
        X[i] = f_x(X[i-1], ctrl_input[i-1], mu_f, mu_r, dt) + w()
        
        alpha_f, alpha_r = f_alpha(X[i], ctrl_input[i])
        mu_f = mu_y(alpha_f)
        mu_r = mu_y(alpha_r)
        Y[i] = f_y(X[i], ctrl_input[i], mu_f, mu_r) + e()
    
    return X, Y



#### This section defines a function to run the online version of the algorithm

def Vehicle_APF(Y):
    
    #variables for logging
    Sigma_X = np.zeros((steps,N_particles,2))
    Sigma_Y = np.zeros((steps,N_particles,2))
    Sigma_mu_f = np.zeros((steps,N_particles))
    Sigma_alpha_f = np.zeros((steps,N_particles))
    Sigma_mu_r = np.zeros((steps,N_particles))
    Sigma_alpha_r = np.zeros((steps,N_particles))

    # front tire
    Mean_f = np.zeros((steps, 1, N_basis_fcn))
    Col_Cov_f = np.zeros((steps, N_basis_fcn, N_basis_fcn))
    Row_Scale_f = np.zeros((steps, 1, 1))
    df_f = np.zeros((steps,))

    # rear tire
    Mean_r = np.zeros((steps, 1, N_basis_fcn))
    Col_Cov_r = np.zeros((steps, N_basis_fcn, N_basis_fcn))
    Row_Scale_r = np.zeros((steps, 1, 1))
    df_r = np.zeros((steps,))
    
    # parameters for the sufficient statistics
    GP_stats_f = [
        np.zeros((N_particles, N_basis_fcn, 1)),
        np.zeros((N_particles, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_particles, 1, 1)),
        np.zeros((N_particles,))
    ]
    GP_stats_r = [
        np.zeros((N_particles, N_basis_fcn, 1)),
        np.zeros((N_particles, N_basis_fcn, N_basis_fcn)),
        np.zeros((N_particles, 1, 1)),
        np.zeros((N_particles,))
    ]
    
    # initial values for states
    Sigma_X[0,...] = np.random.multivariate_normal(x0, P0, (N_particles,))
    Sigma_mu_f[0,...] = np.random.normal(0, 1e-2, (N_particles,))
    Sigma_mu_r[0,...] = np.random.normal(0, 1e-2, (N_particles,))
    weights = np.ones((steps,N_particles))/N_particles
    
    # update GP
    phi_f = jax.vmap(
        functools.partial(features_MTF_front, u=ctrl_input[0])
        )(Sigma_X[0])
    phi_r = jax.vmap(
        functools.partial(features_MTF_rear, u=ctrl_input[0])
        )(Sigma_X[0])
    GP_stats_f = list(jax.vmap(prior_mniw_calcStatistics)(
        Sigma_mu_f[0,...],
        phi_f
    ))
    GP_stats_r = list(jax.vmap(prior_mniw_calcStatistics)(
        Sigma_mu_r[0,...],
        phi_r
    ))
    
    
    
    for i in tqdm(range(1,steps), desc="Running Online Algorithm"):
        
        ### Step 1: Propagate GP parameters in time
        
        # apply forgetting operator to statistics for t-1 -> t
        GP_stats_f[0] *= forget_factor
        GP_stats_f[1] *= forget_factor
        GP_stats_f[2] *= forget_factor
        GP_stats_f[3] *= forget_factor
        GP_stats_r[0] *= forget_factor
        GP_stats_r[1] *= forget_factor
        GP_stats_r[2] *= forget_factor
        GP_stats_r[3] *= forget_factor
            
        # calculate parameters of GP from prior and sufficient statistics
        GP_para_f = list(jax.vmap(prior_mniw_2naturalPara_inv)(
            GP_prior_f[0] + GP_stats_f[0],
            GP_prior_f[1] + GP_stats_f[1],
            GP_prior_f[2] + GP_stats_f[2],
            GP_prior_f[3] + GP_stats_f[3]
        ))
        GP_para_r = list(jax.vmap(prior_mniw_2naturalPara_inv)(
            GP_prior_r[0] + GP_stats_r[0],
            GP_prior_r[1] + GP_stats_r[1],
            GP_prior_r[2] + GP_stats_r[2],
            GP_prior_r[3] + GP_stats_r[3]
        ))
            
            
            
        ### Step 2: According to the algorithm of the auxiliary PF, resample 
        # particles according to the first stage weights
        
        # create auxiliary variable for state x
        x_aux = jax.vmap(functools.partial(f_x, u=ctrl_input[i-1], dt=dt))(
            x=Sigma_X[i-1],
            mu_yf=Sigma_mu_f[i-1], 
            mu_yr=Sigma_mu_r[i-1]
        )
        
        # create auxiliary variable for mu front
        phi_f0 = jax.vmap(
            functools.partial(features_MTF_front, u=ctrl_input[i-1])
            )(Sigma_X[i-1])
        phi_f1 = jax.vmap(
            functools.partial(features_MTF_front, u=ctrl_input[i])
            )(x_aux)
        mu_f_aux = jax.vmap(
            functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
            )(
                mean=GP_para_f[0],
                col_cov=GP_para_f[1],
                row_scale=GP_para_f[2],
                df=GP_para_f[3],
                y1=Sigma_mu_f[i-1],
                basis1=phi_f0,
                basis2=phi_f1
        )[0]
        
        # create auxiliary variable for mu rear
        phi_r0 = jax.vmap(
            functools.partial(features_MTF_rear, u=ctrl_input[i-1])
            )(Sigma_X[i-1])
        phi_r1 = jax.vmap(
            functools.partial(features_MTF_rear, u=ctrl_input[i])
            )(x_aux)
        mu_r_aux = jax.vmap(
            functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
            )(
                mean=GP_para_r[0],
                col_cov=GP_para_r[1],
                row_scale=GP_para_r[2],
                df=GP_para_r[3],
                y1=Sigma_mu_r[i-1],
                basis1=phi_r0,
                basis2=phi_r1
        )[0]
        
        # calculate first stage weights
        y_aux = jax.vmap(
            functools.partial(f_y, u=ctrl_input[i])
            )(
                x=x_aux,
                mu_yf=mu_f_aux, 
                mu_yr=mu_r_aux)
        l_y_aux = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(y_aux)
        p = weights[i-1] * np.exp(l_y_aux)
        p = p/np.sum(p)
        
        #abort
        if np.any(np.isnan(p)):
            print("Particle degeneration at auxiliary weights")
            break
        
        # draw new indices
        u = np.random.rand()
        idx = np.array(systematic_SISR(u, p))
        idx[idx >= N_particles] = N_particles - 1 # correct out of bounds indices from numerical errors
        
        
        
        ### Step 3: Make a proposal by generating samples from the hirachical 
        # model
        
        # sample from proposal for x at time t
        w_x = w((N_particles,))
        Sigma_X[i] = jax.vmap(
            functools.partial(f_x, u=ctrl_input[i-1], dt=dt)
            )(
                x=Sigma_X[i-1,idx],
                mu_yf=Sigma_mu_f[i-1,idx],
                mu_yr=Sigma_mu_r[i-1,idx]
                ) + w_x
        
        ## sample proposal for mu front at time t
        # evaluate basis functions
        phi_f0 = jax.vmap(
            functools.partial(features_MTF_front, u=ctrl_input[i-1])
            )(Sigma_X[i-1,idx])
        phi_f1 = jax.vmap(
            functools.partial(features_MTF_front, u=ctrl_input[i])
            )(Sigma_X[i])
        
        # calculate conditional predictive distribution
        c_mean, c_col_scale, c_row_scale, df = jax.vmap(
            functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
            )(
                mean=GP_para_f[0][idx],
                col_cov=GP_para_f[1][idx],
                row_scale=GP_para_f[2][idx],
                df=GP_para_f[3][idx],
                y1=Sigma_mu_f[i-1,idx],
                basis1=phi_f0,
                basis2=phi_f1
        )
            
        # generate samples
        c_col_scale_chol = np.sqrt(np.squeeze(c_col_scale))
        c_row_scale_chol = np.sqrt(np.squeeze(c_row_scale))
        t_samples = np.random.standard_t(df=df)
        Sigma_mu_f[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
        
        ## sample proposal for mu rear at time t
        # evaluate basis fucntions
        phi_r0 = jax.vmap(
            functools.partial(features_MTF_rear, u=ctrl_input[i-1])
            )(Sigma_X[i-1,idx])
        phi_r1 = jax.vmap(
            functools.partial(features_MTF_rear, u=ctrl_input[i])
            )(Sigma_X[i])
        
        # calculate conditional predictive distribution
        c_mean, c_col_scale, c_row_scale, df = jax.vmap(
            functools.partial(prior_mniw_CondPredictive, y1_var=3e-2)
            )(
                mean=GP_para_r[0][idx],
                col_cov=GP_para_r[1][idx],
                row_scale=GP_para_r[2][idx],
                df=GP_para_r[3][idx],
                y1=Sigma_mu_r[i-1,idx],
                basis1=phi_r0,
                basis2=phi_r1
        )
            
        # generate samples
        c_col_scale_chol = np.sqrt(np.squeeze(c_col_scale))
        c_row_scale_chol = np.sqrt(np.squeeze(c_row_scale))
        t_samples = np.random.standard_t(df=df)
        Sigma_mu_r[i] = c_mean + c_col_scale_chol * t_samples * c_row_scale_chol
            
            
        
        # Update the sufficient statistics of GP with new proposal
        T_0, T_1, T_2, T_3 = list(jax.vmap(prior_mniw_calcStatistics)(
            Sigma_mu_f[i],
            phi_f1
        ))
        GP_stats_f[0] = GP_stats_f[0][idx] + T_0
        GP_stats_f[1] = GP_stats_f[1][idx] + T_1
        GP_stats_f[2] = GP_stats_f[2][idx] + T_2
        GP_stats_f[3] = GP_stats_f[3][idx] + T_3
        
        T_0, T_1, T_2, T_3 = list(jax.vmap(prior_mniw_calcStatistics)(
            Sigma_mu_r[i],
            phi_r1
        ))
        GP_stats_r[0] = GP_stats_r[0][idx] + T_0
        GP_stats_r[1] = GP_stats_r[1][idx] + T_1
        GP_stats_r[2] = GP_stats_r[2][idx] + T_2
        GP_stats_r[3] = GP_stats_r[3][idx] + T_3
        
        
        
        # calculate new weights
        sigma_y = jax.vmap(
            functools.partial(f_y, u=ctrl_input[i])
            )(
                x=Sigma_X[i],
                mu_yf=Sigma_mu_f[i], 
                mu_yr=Sigma_mu_r[i])
        Sigma_Y[i] = sigma_y[:,:2]
        l_y = jax.vmap(functools.partial(log_likelihood_Normal, mean=Y[i], cov=R))(sigma_y)
        weights[i] = np.exp(l_y - l_y_aux[idx])
        weights[i] = weights[i]/np.sum(weights[i])
        
        #abort
        if np.any(np.isnan(weights[i])):
            print("Particle degeneration at new weights")
            break
        
        
        
        # logging
        GP_para_logging = prior_mniw_2naturalPara_inv(
            GP_prior_f[0] + np.einsum('n...,n->...', GP_stats_f[0], weights[i]),
            GP_prior_f[1] + np.einsum('n...,n->...', GP_stats_f[1], weights[i]),
            GP_prior_f[2] + np.einsum('n...,n->...', GP_stats_f[2], weights[i]),
            GP_prior_f[3] + np.einsum('n...,n->...', GP_stats_f[3], weights[i])
        )
        Mean_f[i] = GP_para_logging[0]
        Col_Cov_f[i] = GP_para_logging[1]
        Row_Scale_f[i] = GP_para_logging[2]
        df_f[i] = GP_para_logging[3]
        
        GP_para_logging = prior_mniw_2naturalPara_inv(
            GP_prior_r[0] + np.einsum('n...,n->...', GP_stats_r[0], weights[i]),
            GP_prior_r[1] + np.einsum('n...,n->...', GP_stats_r[1], weights[i]),
            GP_prior_r[2] + np.einsum('n...,n->...', GP_stats_r[2], weights[i]),
            GP_prior_r[3] + np.einsum('n...,n->...', GP_stats_r[3], weights[i])
        )
        Mean_r[i] = GP_para_logging[0]
        Col_Cov_r[i] = GP_para_logging[1]
        Row_Scale_r[i] = GP_para_logging[2]
        df_r[i] = GP_para_logging[3]
        
        Sigma_alpha_f[i], Sigma_alpha_r[i] = jax.vmap(
            functools.partial(f_alpha, u=ctrl_input[i])
            )(
            x=Sigma_X[i]
        )
        
    return Sigma_X, Sigma_mu_r, Sigma_mu_f, Sigma_alpha_f, Sigma_alpha_r, weights, Mean_f, Col_Cov_f, Row_Scale_f, df_f, Mean_r, Col_Cov_r, Row_Scale_r, df_r