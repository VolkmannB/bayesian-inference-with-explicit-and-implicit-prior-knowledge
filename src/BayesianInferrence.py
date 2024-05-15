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
def prior_mniw_2naturalPara(M, V, Psi, nu):
    
    M = jnp.atleast_2d(M)
    Psi = jnp.atleast_2d(Psi)

    V_chol = jnp.linalg.cholesky(V)
    temp = jsc.linalg.cho_solve((V_chol, True), jnp.hstack([M.T, jnp.eye(V.shape[0])]))
    eta_0 = temp[:,:M.shape[0]]
    eta_1 = temp[:,M.shape[0]:]
    eta_2 = M @ eta_0 + Psi
    
    return eta_0, eta_1, eta_2, nu

@jax.jit
def prior_mniw_2naturalPara_inv(eta_0, eta_1, eta_2, eta_3):
    
    eta_1_chol = jnp.linalg.cholesky(eta_1)
    temp = jsc.linalg.cho_solve((eta_1_chol, True), jnp.hstack([eta_0, jnp.eye(eta_1.shape[0])]))
    
    M = temp[:,:eta_0.shape[1]].T
    V = temp[:,eta_0.shape[1]:]
    Psi = eta_2 - M @ eta_0
    
    return jnp.atleast_2d(M), V, jnp.atleast_2d(Psi), eta_3

@jax.jit
def prior_mniw_updateStatistics(T_0, T_1, T_2, T_3, y, phi):
    
    T_0 = T_0 + jnp.outer(phi,y) 
    T_1 = T_1 + jnp.outer(phi,phi)
    T_2 = T_2 + jnp.outer(y,y)
    T_3 = T_3 + 1
    
    return T_0, T_1, T_2, T_3

@jax.jit
def prior_mniw_sampleLikelihood(key, M, V, Psi, nu, phi):
    
    # Calculate parameters of the NIW predictive distribution
    df = nu + 1
    l = 1/(phi @ V @ phi)
    Scale = Psi * (l + 1)/(l * df)
    Scale_chol = jnp.linalg.cholesky(Scale)
    Mean = M @ phi
    
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
def gaussian_RBF(x, inducing_points, lengthscale=1):
    
    r = sq_dist(
        x/lengthscale, 
        inducing_points/lengthscale
        )
        
    return jnp.exp(-0.5*r)



@jax.jit
def bump_RBF(x, inducing_points, lengthscale=1, radius=2):
    
    sq_d = sq_dist(
        x/lengthscale, 
        inducing_points/lengthscale
        )
    
    return jnp.exp(-1 / (1 - jnp.minimum(1, sq_d/radius**2) + 1e-200))



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