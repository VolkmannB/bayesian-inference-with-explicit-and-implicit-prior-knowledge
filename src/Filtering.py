import numpy as np
import numpy.typing as npt
import typing as tp
import itertools
import abc
import jax
import jax.numpy as jnp



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