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



################################################################################
# Priors

# Matrix Normal Inverse Wishart
# For a multivariate Gaussian likelihood with unknown mean and covariance
@jax.jit
def prior_mniw_2naturalPara(M, V, Psi, nu):
    
    M = jnp.atleast_2d(M)
    Psi = jnp.atleast_2d(Psi)

    V_chol = jnp.linalg.cholesky(V)
    temp = jsc.linalg.cho_solve((V_chol, True), jnp.hstack([M.T, jnp.eye(V.shape[0])]))
    eta_1 = temp[:,:M.shape[0]]
    eta_2 = temp[:,M.shape[0]:]
    eta_0 = M @ eta_1 + Psi
    
    return eta_0, eta_1, eta_2, nu

@jax.jit
def prior_mniw_2naturalPara_inv(eta_0, eta_1, eta_2, eta_3):
    
    eta_2_chol = jnp.linalg.cholesky(eta_2)
    temp = jsc.linalg.cho_solve((eta_2_chol, True), jnp.hstack([eta_1, jnp.eye(eta_2.shape[0])]))
    
    M = temp[:,:eta_1.shape[1]].T
    V = temp[:,eta_1.shape[1]:]
    Psi = eta_0 - M @ eta_1
    
    return jnp.atleast_2d(M), V, jnp.atleast_2d(Psi), eta_3

@jax.jit
def prior_mniw_updateStatistics(T_0, T_1, T_2, T_3, y, psi):
    
    T_0 = T_0 + jnp.outer(y,y)
    T_1 = T_1 + jnp.outer(psi,y)
    T_2 = T_2 + jnp.outer(psi,psi)
    T_3 = T_3 + 1
    
    return T_0, T_1, T_2, T_3

@jax.jit
def prior_mniw_samplePredictiveDist(eta, T, psi, key):
    
    # calculate standard posterior parameters from natural parameters and 
    # sufficient statistics
    eta_0 = eta[0] + T[0]
    eta_1 = eta[1] + T[1]
    eta_2 = eta[2] + T[2]
    eta_3 = eta[3] + T[3]
    M, V, Psi, nu = prior_iw_2naturalPara_inv(eta_0, eta_1, eta_2, eta_3)
    
    # Calculate parameters of the NIW predictive distribution
    Scale = Psi * (psi @ V @ psi)
    Scale_chol = jnp.linalg.cholesky(Scale)
    Mean = M @ psi
    df = eta_3 + 1
    
    # generate a sample of the carrier measure wich is a t distribution
    sample = jax.random.t(key, df, (M.shape[0],))
    
    return Mean + Scale_chol @ sample
    



# Inverse Wishart
# For a multivariate Gaussian likelihood with known mean and unknown covariance
@jax.jit
def prior_iw_2naturalPara(Psi, nu):
    
    return Psi, nu

@jax.jit
def prior_iw_2naturalPara_inv(eta_0, eta_1):

    return eta_0, eta_1

@jax.jit
def prior_iw_updateStatistics(T_0, T_1, y, mu):
    
    e = y - mu
    T_0 = T_0 + jnp.outer(e,e)
    T_1 = T_1 + 1
    
    return T_0, T_1


def sample_nig(mean, Lambda_inv, a, b, key, N=1):
    
    V_chol = jnp.linalg.cholesky(b/a*Lambda_inv)
    
    T = jax.random.t(key, 2*a, (N, Lambda_inv.shape[0]))
    
    return jnp.squeeze(mean + jnp.einsum('ij,...j->...i', V_chol, T))



def sample_mniw(M, Lambda, V, nu, psi, key):
    
    n_x = M.shape[1]
    M_ = psi @ M
    nu_ = nu - n_x + 1
    Lambda_chol = jnp.linalg.cholesky(Lambda)
    scale = psi @ jsc.linalg.cho_solve((Lambda_chol, True), psi)
    
    sample = jax.random.t(key, nu_, (n_x,))
    
    return M_ + jnp.linalg.cholesky(scale*V) @ sample
    


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
    V_n = V_0 + jnp.outer(s, s) + d.T @ Lambda_0 @ d
    
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
    
    M_star = jnp.linalg.solve(Sigma_star_inv, Psi.T).T
    M_bar = M_star @ psi
    
    Lambda_star = Lambda_0 + Phi - M_star @ Psi.T
    
    temp = psi @ jnp.linalg.solve(Sigma_star_inv, psi)
    Lambda_bar = temp * Lambda_star
    Lambda_bar_chol = jnp.linalg.cholesky(Lambda_bar)
    
    w = jax.random.t(key, nu-n_x+1, (n_x,))
    
    return M_bar + Lambda_bar_chol @ w
    


################################################################################
# Basis Function



@jax.jit
def gaussian_RBF(x, inducing_points, lengthscale=1):
    
    r = sq_dist(
        x/lengthscale, 
        inducing_points/lengthscale
        )
        
    return jnp.exp(-0.5*r)