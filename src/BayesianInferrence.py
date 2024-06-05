import numpy as np
import numpy.typing as npt
import typing as tp
import functools
import jax.numpy as jnp
import jax.scipy as jsc
import jax
import functools
import itertools



################################################################################
# Priors

# Matrix Normal Inverse Wishart
# For a multivariate Gaussian likelihood with unknown mean and covariance
@jax.jit
def prior_mniw_2naturalPara(mean, col_cov, row_scale, df):
    
    mean = jnp.atleast_2d(mean)
    row_scale = jnp.atleast_2d(row_scale)

    col_cov_chol = jnp.linalg.cholesky(col_cov)
    temp = jsc.linalg.cho_solve(
        (col_cov_chol, True), 
        jnp.hstack([mean.T, jnp.eye(col_cov.shape[0])])
        )
    
    eta_0 = temp[:,:mean.shape[0]]
    eta_1 = temp[:,mean.shape[0]:]
    eta_2 = mean @ eta_0 + row_scale
    
    return eta_0, eta_1, eta_2, df

@jax.jit
def prior_mniw_2naturalPara_inv(eta_0, eta_1, eta_2, eta_3):
    
    eta_1_chol = jnp.linalg.cholesky(eta_1)
    temp = jsc.linalg.cho_solve(
        (eta_1_chol, True), 
        jnp.hstack([eta_0, jnp.eye(eta_1.shape[0])])
        )
    
    mean = temp[:,:eta_0.shape[1]].T
    col_cov = temp[:,eta_0.shape[1]:]
    row_scale = eta_2 - mean @ eta_0
    
    return jnp.atleast_2d(mean), col_cov, jnp.atleast_2d(row_scale), eta_3

@jax.jit
def prior_mniw_updateStatistics(T_0, T_1, T_2, T_3, y, basis):
    
    T_0 = T_0 + jnp.outer(basis,y) 
    T_1 = T_1 + jnp.outer(basis,basis)
    T_2 = T_2 + jnp.outer(y,y)
    T_3 = T_3 + 1
    
    return T_0, T_1, T_2, T_3


@jax.jit
def prior_mniw_calcStatistics(eta_0, eta_1, eta_2, eta_3, old_data, new_data):
    
    eta_0 += jnp.outer(new_data,new_data) 
    eta_1 += jnp.outer(old_data,new_data)
    eta_2 += jnp.outer(old_data,old_data)
    eta_3 += new_data.shape[1]
    
    return eta_0, eta_1, eta_2, eta_3

@jax.jit
def prior_mniw_Predictive(mean, col_cov, row_scale, df, basis):
    
    basis = jnp.atleast_2d(basis)
    col_cov = jnp.atleast_2d(col_cov)
    row_scale = jnp.atleast_2d(row_scale)
    
    n_b = basis.shape[0]
    
    # degrees of freedom
    df = df + 1
    
    # mean
    mean = jnp.squeeze(basis @ mean.T)
    
    # column variance
    col_scale = basis @ col_cov @ basis.T + jnp.eye(n_b)
    
    # conditional scale matrix of predictive distribution
    row_scale = row_scale / df
    
    return mean, col_scale, row_scale, df

@jax.jit
def prior_mniw_CondPredictive(mean, col_cov, row_scale, df, y1, y1_var, basis1, basis2):
    
    basis1 = jnp.atleast_2d(basis1)
    basis2 = jnp.atleast_2d(basis2)
    col_cov = jnp.atleast_2d(col_cov)
    row_scale = jnp.atleast_2d(row_scale)
    
    n_b1 = basis1.shape[0]
    n_b2 = basis2.shape[0]
    
    # degrees of freedom
    df = df + 1
    
    # entries of col covariance in function space
    col_cov_11 = basis1 @ col_cov @ basis1.T + jnp.eye(n_b1)*y1_var
    col_cov_22 = basis2 @ col_cov @ basis2.T + jnp.eye(n_b2)
    col_cov_12 = basis1 @ col_cov @ basis2.T
    
    # prediction error
    err = y1 - basis1 @ mean.T
    
    # gain
    col_cov_11_chol = jnp.linalg.cholesky(col_cov_11)
    G = jsc.linalg.cho_solve((col_cov_11_chol, True), col_cov_12)
    
    # conditional mean
    c_mean = jnp.squeeze(basis2 @ mean.T + err*G)
    
    # conditional column variance
    c_col_scale = col_cov_22 - col_cov_12.T*G
    
    # conditional scale matrix of predictive distribution
    c_row_scale = row_scale / df
    
    return c_mean, c_col_scale, c_row_scale, df
    


# Normal Inverse Wishart
@jax.jit
def prior_niw_calcStatistics(eta_0, eta_1, eta_2, eta_3, new_data):
    
    eta_0 += jnp.outer(new_data,new_data) 
    eta_1 += jnp.sum(new_data,axis=1)
    eta_2 += new_data.shape[1]
    eta_3 += new_data.shape[1]
    
    return eta_0, eta_1, eta_2, eta_3


@jax.jit
def prior_niw_2naturalPara(mean, normal_scale, iw_scale, df):
    
    mean = jnp.atleast_2d(mean)
    iw_scale = jnp.atleast_2d(iw_scale)

    eta_1 = mean/normal_scale
    eta_2 = 1/normal_scale
    eta_3 = df
    eta_0 = jnp.outer(eta_1,mean) + iw_scale

    return eta_0, eta_1, eta_2, eta_3 


@jax.jit
def prior_niw_2naturalPara_inv(eta_0, eta_1, eta_2, eta_3):
    
    mean = eta_1/eta_2
    normal_scale = 1/eta_2
    iw_scale = eta_0 - jnp.outer(mean,eta_1)
    df = eta_3
    
    return jnp.atleast_2d(mean), normal_scale, jnp.atleast_2d(iw_scale), df




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
    
@jax.jit
def prior_iw_calcStatistics(T_0, T_1, mu, y):
    
    e = y - mu
    T_0 = T_0 + jnp.outer(e,e)
    T_1 = T_1 + y.shape[1]
    
    return T_0, T_1



################################################################################
# Basis Function
    


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
def gaussian_RBF(x, loc, lengthscale=1):
    
    sq_r = sq_dist(x/lengthscale, loc/lengthscale)
        
    return jnp.exp(-0.5*sq_r)



@jax.jit
def bump_RBF(x, loc, lengthscale=1, radius=2):
    
    sq_r = sq_dist(x/lengthscale, loc/lengthscale)
    sq_r = jnp.minimum(1, sq_r/radius**2)
    
    return jnp.exp(-1 / (1 - sq_r + 1e-200))



@jax.jit
def C2_InversePoly_RBF(x, loc, lengthscale=1, radius=2):
    
    r = jnp.sqrt(sq_dist(x/lengthscale, loc/lengthscale))
    r = jnp.minimum(1, r/radius)
    
    return 1 + 8/(1+r)**3 - 6/(1+r)**4



def generate_Hilbert_BasisFunction(M, L, l, sigma, j_start=1, j_step=1):
    
    # dimensionality of the input
    L = np.atleast_2d(L)
    d = L.shape[0]
    
    # center domain L
    L_center = (L[:,0] + L[:,1])/2
    
    # set start index to 1 if value is negative
    if j_start < 1:
        j_start = 1
    
    # set indices per dimension
    L_size = L[:,1] - L[:,0]
    j_end = M*j_step+1+j_start
    j = np.arange(j_start, j_end, j_step)
    
    # create all combinations of indices
    S = np.array(list(
        itertools.product(*np.repeat(j[None,:], d, axis=0))
        ))
    
    # calculate eigenvalues
    eig_val = (np.pi*S/L_size)**2
    
    # sort eigenvalues in decending order
    idx = np.flip(np.argsort(np.sum(eig_val, axis=1)))
    
    # get M combinations of highest eigenvalues
    idx = idx[:M]
    
    # create function for basis functions
    eigen_fun = lambda x: functools.partial(_eigen_fnc, L=L_size/2, eigen_val=eig_val[idx])(x=x-L_center)
    
    # calculate spectral density
    sd = _spectral_density_Gaussian(sigma, np.atleast_1d(l), np.sqrt(eig_val[idx]))
    
    return jax.jit(eigen_fun), sd
    
def _eigen_fnc(x, eigen_val, L):
    
    return jnp.prod(jnp.sqrt(1/L) * jnp.sin(jnp.sqrt(eigen_val) * (x + L)), axis=1)

def _spectral_density_Gaussian(alpha, l, omega):
    
    d = omega.shape[1]
    r = omega*l
    
    if len(l) == 1:
        l_d = l**d
    else:
        l_d = np.prod(l)
    
    s = alpha * np.sqrt(2*np.pi)**d * l_d * np.exp(-0.5 * np.sum(r**2, axis=1))
    
    return s