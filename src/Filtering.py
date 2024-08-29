import numpy as np
import numpy.typing as npt
import typing as tp
import itertools
import abc
import jax
import jax.numpy as jnp
import jax.scipy as jsc



@jax.jit
def systematic_SISR(u, w: npt.ArrayLike):
    
    # number of samples
    N = len(w)
    
    # select deterministic samples
    U = u/N + jnp.array(range(0, N))/N
    W = jnp.cumsum(w, 0)
    
    indices = jnp.searchsorted(W, U)
    
    return indices



@jax.jit
def squared_error(x, y, cov):
    
    cov = jnp.atleast_2d(cov)
    dx = jnp.atleast_1d(x) - jnp.atleast_1d(y)
    
    t = jnp.linalg.solve(cov, dx)
    r = dx @ t
    
    return jnp.exp(-0.5 * r)



@jax.jit
def logweighting(x, y, cov):
    
    cov = jnp.atleast_2d(cov)
    dx = jnp.atleast_1d(x) - jnp.atleast_1d(y)
    
    t = jnp.linalg.solve(cov, dx)
    r = dx @ t
    
    return -0.5 * r



@jax.jit
def EnKF_update(sigma_x, sigma_y, y, R):
    
    N = sigma_x.shape[0]
    
    mean_y = jnp.mean(sigma_y, axis=0)
    mean_x = jnp.mean(sigma_x, axis=0)
    
    x_centered = sigma_x - mean_x
    y_centered = sigma_y - mean_y
    
    P_yy = 1/(N-1) * jnp.einsum('ji,jk->ik', y_centered, y_centered) + R
    P_xy = 1/(N-1) * jnp.einsum('ji,jk->ik', x_centered, y_centered)
    
    K_T = jnp.linalg.solve(P_yy, P_xy.T)
    
    return sigma_x + (y - sigma_y) @ K_T



@jax.jit
def log_likelihood_Normal(observed, mean, cov):
    """
    Computes the un-normalized log likelihood for the multivariate
    normal distribution.
    
    .. math::
    \log\mathcal{N}(x|mean, cov) \propto -\frac{1}{2} (x-mean)^\mathrm{T} cov (x-mean)

    Args:
        x (ArrayLike): input
        mean (ArrayLike): mean vector
        cov (ArrayLike): Covariance martrix

    Returns:
        ArrayLike: log-likelihood
    """
    
    cov = jnp.atleast_2d(cov)
    dx = jnp.atleast_1d(observed) - jnp.atleast_1d(mean)
    
    L = jnp.linalg.cholesky(cov)
    t = jsc.linalg.cho_solve((L, True), dx)
    r = dx @ t
    
    return -0.5 * r



@jax.jit
def log_likelihood_Multivariate_t(observed, mean, scale, df):
    """
    Computes the unnormalized log-likelihood of a multivariate t-distribution.

    Args:
        observed (array-like): Observation vector of shape (p,).
        mean (array-like): Mean vector of shape (p,).
        scale (array-like): Scale matrix of shape (p, p).
        df (float): Degrees of freedom.

    Returns:
        float: The unnormalized log-likelihood of the observation x.
    """
    
    # Calculate the dimensionality
    observed = jnp.atleast_1d(observed)
    mean = jnp.atleast_1d(mean)
    p = observed.shape[0]
    
    # Calculate the term (x - mu)
    diff = observed - mean
    
    # Compute the Cholesky decomposition of scae matrix
    scale = jnp.atleast_2d(scale)
    L = jnp.linalg.cholesky(scale)
    
    # Compute the squared Mahalanobis distance
    m = jsc.linalg.cho_solve((L, True), diff)
    m_squared = diff @ m
    
    # Calculate the unnormalized log-likelihood
    log_likelihood = -0.5 * (df + p) * jnp.log(1 + (1 / df) * m_squared)
    
    return log_likelihood



def reconstruct_trajectory(Particles, ancestry, idx):
    
    Particles = np.atleast_3d(Particles)
    
    n_steps = Particles.shape[0]
    n_dim = Particles.shape[-1]
    traj = np.zeros((n_steps, n_dim))
    
    ancestor_idx = np.zeros((n_steps,))
    ancestor_idx[-1] = idx
    
    traj[-1] = Particles[-1, idx]
    for i in range(n_steps-2, -1, -1): # run backward in time
        
        ancestor_idx[i] = ancestry[i, int(ancestor_idx[i+1])]
        traj[i] = Particles[i, int(ancestor_idx[i])]
        
    return np.squeeze(traj)