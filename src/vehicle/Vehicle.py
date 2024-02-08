import numpy as np
import math
import jax
import jax.numpy as jnp
import functools



# x = [dpsi, v_y]
# u = [delta, v_x]

##### Simulation

# tire load
def f_Fz(m, g, l_f, l_r):
    
    l = l_f + l_r
    F_zf = m*g*l_r/l
    F_zr = m*g*l_f/l
    return F_zf, F_zr



# friction MTF curve
def mu_y(alpha, mu, B, C, E):
    
    return mu * jnp.sin(C * jnp.arctan(B*(1-E)*jnp.tan(alpha) + E*jnp.arctan(B*jnp.tan(alpha))))



# side slip
def f_alpha(x, u):

    vx_f = u[1]
    vy_f = x[1]
    
    vx_r = u[1]
    vy_r = x[1]
    
    return u[0]-jnp.arctan2(vx_f, vy_f), jnp.arctan2(vx_r, vy_r)


# state space model
def f_dx_sim(x, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r):
    
    F_zf, F_zr = f_Fz(m, g, l_f, l_r)
    alpha_f, alpha_r = f_alpha(x, u)
    mu_yf = mu_y(alpha_f, mu, B_f, C_f, E_f)
    mu_yr = mu_y(alpha_r, mu, B_r, C_r, E_r)
    
    dv_y = 1/m*(F_zf*mu_yf*jnp.cos(u[0]) + F_zr*mu_yr + F_zf*mu_x*jnp.sin(u[0])) - u[1]*x[0]
    dpsi = 1/I_zz*(l_f*F_zf*mu_yf*jnp.cos(u[0]) - l_r*F_zr*mu_yr + l_f*F_zf*mu_x*jnp.sin(u[0]))
    
    return jnp.stack([dv_y, dpsi])



# time discrete state space pdel with Runge-Kutta-4
@jax.jit
def f_x(x, u, dt, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r):
    
    k1 = f_dx_sim(x, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k2 = f_dx_sim(x + dt*k1/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k3 = f_dx_sim(x + dt*k2/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k4 = f_dx_sim(x + dt*k3/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    
    return x + dt/6.0*(k1+2*k2+2*k3+k4)



##### Parameters (dt, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
default_para = jnp.array([
    0.01, # dt
    1720, # m
    1827.5431059723351, # J_z
    1.1625348837209302, # l_f
    1.4684651162790696, #l_r
    9.81, # g
    0.9, # mu_x
    0.9, # mu
    10.0, # B_f
    1.9, # C_f
    0.97, # E_f
    10.0, # B_r
    1.9, # C_r
    0.97, # E_r
])



##### MCMC model for parameter identification
@jax.jit
def f_x_MCMC(x_MCMC, u_MCMC, para, dt, m, I_zz, l_f, l_r, g):
    
    x = x_MCMC[:-1]
    u = jnp.array([x_MCMC[-1], *u_MCMC])
    
    mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r = para.flatten()
    
    k1 = f_dx_sim(x, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k2 = f_dx_sim(x + dt*k1/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k3 = f_dx_sim(x + dt*k2/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    k4 = f_dx_sim(x + dt*k3/2.0, u, m, I_zz, l_f, l_r, g, mu_x, mu, B_f, C_f, E_f, B_r, C_r, E_r)
    
    x_out = x + dt/6.0*(k1+2*k2+2*k3+k4)
    
    return jnp.array([*x_out, x_MCMC[-1]]).flatten()



# bootstrap proposal
@jax.jit
def p_x_Bootstrap(X_MCMC, u_MCMC, para, key, dt, m, I_zz, l_f, l_r, g, Q):
    
    X_MCMC_new = jax.vmap(
        functools.partial(f_x_MCMC, u_MCMC=u_MCMC, para=para, dt=dt, m=m, I_zz=I_zz, l_f=l_f, l_r=l_r, g=g)
        )(
            X_MCMC
        )
    
    return X_MCMC_new + jax.random.multivariate_normal(key, jnp.zeros((Q.shape[0],)), Q, (X_MCMC_new.shape[0],))



# likelihood
@jax.jit
def p_y_Bootstrap(X_MCMC, y, R):
    
    Y_ = X_MCMC[:,:-1]
    
    r = Y_ - y
    
    likelihood = lambda r: jnp.exp(-0.5 * r @ jnp.linalg.solve(R, r))
    
    return jax.vmap(likelihood)(r)



@jax.jit
def systematic_resampling(w, key):
    
    # number of samples
    N = len(w)
    
    # initialize array of indices
    indices = jnp.zeros((N,), dtype=jnp.int32)
    
    # select deterministic samples
    U = jax.random.uniform(key, minval=0, maxval=1/N) + jnp.arange(0, N) / N
    W = jnp.cumsum(w, 0)
    
    indices = jnp.searchsorted(W, U)
    
    return indices



def forward_Bootstrap_PF(Y, U, X_0, para, Q, R):
    
    # generate initial key
    key = jax.random.PRNGKey(np.random.random_integers(100, 1e3))
    
    # variables for logging
    unnormalized_weights = np.zeros((Y.shape[0], X_0.shape[0]))
    X_trajectory = np.zeros((Y.shape[0], *X_0.shape))
    
    # weight initial particles
    unnormalized_weights[0,:] = p_y_Bootstrap(X_0, Y[0,:], R)
    
    # prepare proposal distribution
    proposal_dist = lambda X, u, key: p_x_Bootstrap(X, u, para, key, *default_para[:6], Q)
    
    for t in np.arange(1, Y.shape[0]):
        
        # resample previous particles
        key, temp_key = jax.random.split(key)
        w = unnormalized_weights[t-1,:] / np.sum(unnormalized_weights[t-1,:])
        idx = systematic_resampling(w, temp_key)
        
        # draw samples from the propsal
        key, temp_key = jax.random.split(key)
        resampled_particles = X_trajectory[t-1, idx, :]
        X_trajectory[t,...] = proposal_dist(resampled_particles, U[t-1], temp_key)
        
        # weight particles
        unnormalized_weights[t,:] = p_y_Bootstrap(X_trajectory[t,...], Y[t,:], R)
        
        if np.isclose(0, np.sum(unnormalized_weights[t,:])):
            raise ValueError(f"Particle depleteion at iteration {t}")
    
    return X_trajectory, unnormalized_weights