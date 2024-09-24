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

    temp = jnp.linalg.solve(
        col_cov, 
        jnp.hstack([mean.T, jnp.eye(col_cov.shape[0])])
    )
    
    eta_0 = temp[:,:mean.shape[0]]
    eta_1 = temp[:,mean.shape[0]:]
    eta_2 = mean @ eta_0 + row_scale
    
    eta_3 = df + col_cov.shape[0] + row_scale.shape[0] + 1
    
    return eta_0, eta_1, eta_2, eta_3

@jax.jit
def prior_mniw_2naturalPara_inv(eta_0, eta_1, eta_2, eta_3):
    
    temp = jnp.linalg.solve(
        eta_1, 
        jnp.hstack([eta_0, jnp.eye(eta_1.shape[0])])
    )
    
    mean = temp[:,:eta_0.shape[1]].T
    col_cov = temp[:,eta_0.shape[1]:]
    row_scale = eta_2 - mean @ eta_0
    df = eta_3 - col_cov.shape[0] - row_scale.shape[0] - 1
    
    return jnp.atleast_2d(mean), col_cov, jnp.atleast_2d(row_scale), df

@jax.jit
def prior_mniw_calcStatistics(y, basis):
    
    T_0 = jnp.outer(basis, y)
    T_1 = jnp.outer(basis, basis)
    T_2 = jnp.outer(y, y)
    T_3 = 1
    
    return T_0, T_1, T_2, T_3

@jax.jit
def prior_mniw_Predictive(mean, col_cov, row_scale, df, basis):
    
    basis = jnp.atleast_2d(basis)
    col_cov = jnp.atleast_2d(col_cov)
    row_scale = jnp.atleast_2d(row_scale)
    
    n_b = basis.shape[0]
    
    # degrees of freedom
    df = df + 1 - row_scale.shape[0]
    
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



@jax.jit
def prior_mniw_log_base_measure(T_0, T_1, T_2, T_3):
    
    n = T_2.shape[0]
    m = T_1.shape[0]
    
    temp_1 = - 0.5 * n * m * jnp.log(2*jnp.pi)
    temp_2 = 0.5 * m * jnp.log(jnp.linalg.det(T_1))
    temp_3 = - 0.5 * (T_3 - n -m - 1) * jnp.log(2)
    temp_4 = - jsc.special.multigammaln((T_3 - n -m - 1)/2, n)

    return temp_1 + temp_2 + temp_3 + temp_4

    

# Multivariate Normal
@jax.jit
def prior_n_calcStatistics(eta_0, eta_1, new_data):
    
    eta_0 += jnp.sum(new_data,axis=1)
    eta_1 += new_data.shape[1]
    
    return eta_0, eta_1

@jax.jit
def prior_n_2naturalPara(mean, sample_number):
    
    mean = jnp.atleast_2d(mean)

    eta_0 = mean*sample_number
    eta_1 = sample_number

    return eta_0, eta_1


@jax.jit
def prior_n_2naturalPara_inv(eta_0, eta_1):
    
    mean = eta_0/eta_1
    sample_number = eta_1
    
    return jnp.atleast_2d(mean), sample_number



# Normal Inverse Wishart
@jax.jit
def prior_niw_calcStatistics(eta_0, eta_1, eta_2, eta_3, new_data):
    
    eta_0 += new_data @ new_data.T # jnp.outer(new_data,new_data) 
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
    T_0 = T_0 + e @ e.T
    T_1 = T_1 + y.shape[1]
    
    return T_0, T_1



################################################################################
# Basis Function
    


def sq_dist(x1, x2) -> npt.NDArray:
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



def generate_Hilbert_BasisFunction(
    num_fcn, 
    domain_boundary, 
    lengthscale, 
    scale, 
    idx_start=1, 
    idx_step=1
    ):
    
    # dimensionality of the input
    domain_boundary = np.atleast_2d(domain_boundary)
    num_dims = domain_boundary.shape[0]
    
    # center domain L
    domain_center = (domain_boundary[:,0] + domain_boundary[:,1])/2
    
    # set start index to 1 if value is negative
    if idx_start < 1:
        idx_start = 1
    
    # set indices per dimension
    domain_size = domain_boundary[:,1] - domain_boundary[:,0]
    idx_end = num_fcn*idx_step+1+idx_start
    j = np.arange(idx_start, idx_end, idx_step)
    
    # create all combinations of indices
    S = np.array(list(
        itertools.product(*np.repeat(j[None,:], num_dims, axis=0))
        ))
    
    # calculate eigenvalues
    eig_val = (np.pi*S/domain_size)**2
    
    # sort eigenvalues in decending order
    idx = np.argsort(np.sum(eig_val, axis=1))
    
    # get M combinations of highest eigenvalues
    idx = idx[:num_fcn]
    eig_val = eig_val[idx]
    
    # callable for basis functions
    eigen_fun = lambda x: functools.partial(_eigen_fnc, L=domain_size/2, eigen_val=eig_val)(x=x-domain_center)
    
    # calculate spectral density
    spectral_density_fcn = functools.partial(
        _spectral_density_Gaussian,
        magnitude=scale,
        lengthscale=lengthscale
        )
    spectral_density = jax.vmap(spectral_density_fcn)(freq=np.sqrt(eig_val))

    return jax.jit(eigen_fun), spectral_density
    
def _eigen_fnc(x, eigen_val, L):
    
    return jnp.prod(
        jnp.sqrt(1/L) * jnp.sin(jnp.sqrt(eigen_val) * (x + L)), 
        axis=1
        )

def _spectral_density_Gaussian(freq, magnitude, lengthscale):
    """
    Calculate the spectral density of the squared exponential kernel with 
    individual lengthscales.

    Args:
        freq (ArrayLike): Frequency vector (1D array).
        magnitude (_type_): Variance parameter.
        lengthscale (ArrayLike): Lengthscales for each dimension (1D array).

    Returns:
        float: Spectral density.
    """
    
    # broadcast to correct shapes
    D = len(freq)
    l = jnp.broadcast_to(lengthscale, jnp.asarray(freq).shape)
    
    term1 = magnitude * (2 * jnp.pi)**(D / 2)
    term2 = jnp.prod(l)
    exponent = -0.5 * jnp.sum((l**2) * (freq**2))
    
    return term1 * term2 * jnp.exp(exponent)