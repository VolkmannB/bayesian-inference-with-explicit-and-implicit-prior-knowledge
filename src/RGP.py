from typing import Any
import numpy as np
import numpy.typing as npt
import abc
import typing as tp
import functools
import jax.numpy as jnp
import jax.scipy as jsc
import jax
import functools



def sample_nig(mean, Lambda_inv, a, b, key, N=1):
    
    V_chol = jnp.linalg.cholesky(b/a*Lambda_inv)
    
    T = jax.random.t(key, 2*a, (N, Lambda_inv.shape[0]))
    
    return jnp.squeeze(mean + jnp.einsum('ij,...j->...i', V_chol, T))



def sq_dist(x1: npt.ArrayLike, x2: npt.ArrayLike) -> npt.NDArray:
    """
    This function calculates the squared euclidean distance ||x||_2^2 between 
    all pairs of vectors in x1 and x2.

    Args:
        x1 (npt.ArrayLike): An [n1, m] array of coordinates.
        x2 (npt.ArrayLike): An [n2, m] array of coordinates.

    Returns:
        npt.NDArray: An [n1, n2] array containing the squared euclidean distance 
        between each pair of vectors.
    """
    
    d = jnp.atleast_2d(x1.T).T[...,jnp.newaxis,:] - jnp.atleast_2d(x2.T).T[jnp.newaxis,:,:]
    distance = (d**2).sum(axis=-1)
    
    return jnp.squeeze(distance)



@jax.jit
def update_nig_prior(mu_0, Lambda_0, alpha_0, beta_0, psi, y):
    """
    Update the normal-inverse gamma (NIG) prior for Bayesian linear regression.

    Parameters:
    - mu_0: Mean vector(s) of shape (n_features,)
    - Lambda_0_inv: Inverse of the precision matrix(s) of shape (n_features, n_features)
    - alpha_0: Shape parameter(s)
    - beta_0: Scale parameter(s)
    - X: Feature matrix of shape (n_samples, n_features) for the new data points
    - y: Response vector of shape (n_samples,) for the new data points

    Returns:
    - mu_n: Updated mean vector(s) of shape (n_features,)
    - Lambda_n_inv: Updated inverse precision matrix(s) of shape (n_features, n_features)
    - alpha_n: Updated shape parameter(s)
    - beta_n: Updated scale parameter(s)
    """
    
    # Update precision matrix
    xTx = jnp.outer(psi, psi)
    Lambda_n = Lambda_0 + xTx

    # Update mean
    xTy = psi * y
    mu_n = jnp.linalg.solve(Lambda_n, xTy + Lambda_0 @ mu_0)

    # Update parameters of inverse gamma
    alpha_n = alpha_0 + 0.5
    mLm_0 = mu_0 @ Lambda_0 @ mu_0
    mLm_n = mu_n @ Lambda_n @ mu_n
    beta_n = beta_0 + 0.5 * (y**2 + mLm_0 - mLm_n)

    return [mu_n, Lambda_n, alpha_n, beta_n]



@jax.jit
def update_normal_prior(mu_0, P_0, psi, y, sigma):
    
        # mean prediction error
        e = y - psi @ mu_0
        
        # covariance matrix of prediction
        P_xy = P_0 @ psi
        P_yy = psi @ P_xy + sigma
        
        # gain matrix
        G = P_xy/P_yy
        
        # k measurments; j basis functions
        mu_n = mu_0 + G * e
        P_n = P_0 - P_yy * jnp.outer(G, G)
        
        return [jnp.squeeze(mu_n), jnp.squeeze(P_n)]



@jax.jit
def update_mniw_prior(M_0, Lambda_0, V_0, nu_0, psi, y):
    
    xTx = jnp.outer(psi, psi)
    Lambda_n = Lambda_0 + xTx
    
    xTy = jnp.outer(psi, y)
    M_n = jnp.linalg.solve(Lambda_n, Lambda_0 @ M_0 + xTy)
    
    s = y - psi @ M_n
    d = M_n - M_0
    V_n = V_0 + np.outer(s, s) + d @ Lambda_0 @ d
    
    nu_n = nu_0 + 1
    
    return [M_n, Lambda_n, V_n, nu_n]



@jax.jit
def update_BMNIW_prior(Phi_0, Psi_0, Sigma_0, nu_0, x_1, psi_0):
    
    Phi_1 = Phi_0 + jnp.outer(x_1, x_1)
    Psi_1 = Psi_0 + jnp.outer(x_1, psi_0)
    Sigma_1 = Sigma_0 + jnp.outer(psi_0, psi_0)
    nu_1 = nu_0 + 1
    
    return [Phi_1, Psi_1, Sigma_1, nu_1]



@jax.jit
def sample_BMNIW_prior(Phi, Psi, Sigma, nu, Lambda_0, v, psi, key):
    
    n_x = Phi.shape[0]
    
    Sigma_star_inv = Sigma + jnp.diag(1/v)
    Sigma_star_inv_chol = jnp.linalg.cholesky(Sigma_star_inv)
    Sigma_star = jsc.linalg.cho_solve((Sigma_star_inv_chol, True), jnp.eye(Sigma.shape[0]))
    Sigma_star_chol = jnp.linalg.cholesky(Sigma_star)
    
    M = jsc.linalg.cho_solve((Sigma_star_inv_chol, True), Psi.T).T
    M_bar = M @ psi
    
    Lambda_star = Lambda_0 + Phi - M @ Psi.T
    # Lambda_star_chol = jnp.linalg.cholesky(Lambda_star)
    
    temp = Sigma_star_chol @ psi
    Lambda_bar = jnp.inner(temp, temp) * Lambda_star
    Lambda_bar_chol = jnp.linalg.cholesky(Lambda_bar)
    
    w = jax.random.t(key, nu-n_x+1, (n_x,))
    
    return M_bar + Lambda_bar_chol @ w
    


################################################################################
# Basis Function



def gaussian_RBF(x, inducing_points, lengthscale=1):
    
    r = sq_dist(
        x/lengthscale, 
        inducing_points/lengthscale
        )
        
    return jnp.exp(-0.5*r)