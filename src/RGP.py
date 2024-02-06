from typing import Any
import numpy as np
import numpy.typing as npt
import abc
import typing as tp
import functools
import jax.numpy as jnp
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
        x1 (npt.ArrayLike): An [..., n_a, m] array of coordinates.
        x2 (npt.ArrayLike): an [..., n_b, m] array of coordinates.

    Returns:
        npt.NDArray: An [..., n_a, n_b] array containing the squared euclidean distance 
        between each pair of vectors.
    """
    
    a = jnp.asarray(x1)
    b = jnp.asarray(x2)
    
    distance = ((x1[..., jnp.newaxis,:] - x2[...,jnp.newaxis,:,:])**2).sum(axis=-1)
    
    return jnp.squeeze(distance)



@jax.jit
def sherman_morrison_inverse(A_inv, x):
    """
    Computes the Sherman-Morrison formula for a matrix A^{-1} and a vector x.

    Args:
        A_inv (numpy.ndarray): Inverse of matrix A.
        x (numpy.ndarray): Vector x.

    Returns:
        numpy.ndarray: Updated inverse (A - x * x^T)^{-1}.

    Raises:
        ValueError: If the denominator in the formula is zero, indicating the formula is not applicable.
    """
    A_inv_x = jnp.dot(A_inv, x)
    denominator = 1.0 + jnp.dot(x, A_inv_x)

    # Check if the denominator is zero to avoid division by zero
    # if denominator == 0:
    #     raise ValueError("Denominator is zero. Sherman-Morrison formula not applicable.")

    update_term = jnp.outer(A_inv_x, A_inv_x) / denominator

    updated_inverse = A_inv - update_term

    return updated_inverse



@jax.jit
def update_nig_prior(mu_0, Lambda_0, Lambda_0_inv, alpha_0, beta_0, psi, y):
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

    # Sherman-Morrison formula for updating inverse precision matrix
    Lambda_n_inv = sherman_morrison_inverse(Lambda_0_inv, psi)

    # Update mean
    xTy = psi * y
    mu_n = Lambda_n_inv @ (xTy + Lambda_0 @ mu_0)

    # Update parameters of inverse gamma
    alpha_n = alpha_0 + 0.5
    mLm_0 = mu_0 @ Lambda_0 @ mu_0
    mLm_n = mu_n @ Lambda_n @ mu_n
    beta_n = beta_0 + 0.5 * (y**2 + mLm_0 - mLm_n)

    # Squeeze dimensions back if necessary
    mu_n = jnp.squeeze(mu_n)
    Lambda_n_inv = jnp.squeeze(Lambda_n_inv)
    alpha_n = jnp.squeeze(alpha_n)
    beta_n = jnp.squeeze(beta_n)

    return mu_n, Lambda_n, Lambda_n_inv, alpha_n, beta_n



@jax.jit
def update_normal_prior(mu_0, P_0, psi, y, sigma, jitter_val):
    
        # mean prediction error
        e = y - psi @ mu_0
        
        # covariance matrix of prediction
        P_xy = P_0 @ psi
        P_yy = psi @ P_xy + sigma + jitter_val
        
        # gain matrix
        G = P_xy/P_yy
        
        # k measurments; j basis functions
        mu_n = mu_0 + G * e
        P_n = P_0 - P_yy * jnp.outer(G, G)
        
        return jnp.squeeze(mu_n), jnp.squeeze(P_n)
    


################################################################################
# Basis Function



def gaussian_RBF(x, inducing_points, lengthscale=1):
    
    r = sq_dist(
        x/lengthscale, 
        inducing_points/lengthscale
        )
        
    return jnp.exp(-0.5*r)



################################################################################
# GP Models    



class BaseGP(abc.ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def predict(self, x: npt.ArrayLike):
        pass
    
    @abc.abstractmethod
    def update(self, x: npt.ArrayLike, y: npt.ArrayLike, var: npt.ArrayLike):
        pass
        

        
        
class EnsambleGP(BaseGP):
    
    def __init__(
        self,
        basis_function: tp.Callable,
        n_basis: int,
        w0: npt.ArrayLike,
        cov0: npt.ArrayLike,
        N: int,
        error_cov: float = 1e-3
        ) -> None:
        
        super().__init__()
        
        if not callable(basis_function):
            raise ValueError("basis_function must be a callable object")
        self.H = basis_function
        
        if len(w0) != n_basis:
            raise ValueError("Number of weights does not match basis functions. Expected {0} but got {1}".format(n_basis, len(w0))) 
        if cov0.shape[0] != cov0.shape[1] or len(cov0.shape) != 2:
            raise ValueError('Covariance matrix must be quadratic')
        if w0.size != cov0.shape[0]:
            raise ValueError('Size of mean vector does not match covariance matrix')
        
        # generate ensamble
        if not isinstance(N, int) or N <= 0:
            raise ValueError('Ensamble size must be a positive integer')
        self.W = w0 + np.random.randn(N,len(w0)) @ np.linalg.cholesky(cov0).T
        
        if error_cov <= 0:
            raise ValueError("Error variance must be positive")
        self.error_cov = error_cov
        
        # tapering matrix
        self.T = np.ones((n_basis, n_basis))
        
    
    
    @property
    def mean(self):
        return self.W.mean(axis=0)
    
    
    
    @property
    def cov(self):
        return np.cov(self.W.T) * self.T
    
    
    
    def predict(self, x: npt.ArrayLike):
        
        x_ = np.asarray(x)
        
        # basis functions
        H = self.H(x_)
        
        return H @ self.W.T + np.sqrt(self.error_cov) * np.random.randn(x_.shape[0], self.W.shape[0])
    
    
    
    def ensample_predict(self, x: npt.ArrayLike):
        
        x_ = np.asarray(x)
        
        # basis functions
        H = self.H(x_)
        
        return np.sum(H*self.W, axis=1)
    
    
    
    def update(self, x: npt.ArrayLike, y: npt.ArrayLike, var: npt.ArrayLike):
        
        # basis functions
        H = self.H(np.asarray(x))
        
        # ensamble of function values
        X = H @ self.W.T
        
        # model and measurment uncertainty
        y_var = np.diag(var)
        f_var = np.diag(np.ones(x.shape[-2])*self.error_cov)
        
        # Kalman gain P_xy/P_yy
        P_ww = self.cov
        P_yy = H @ np.einsum('nm,km->nk', P_ww, H) + y_var + f_var
        K_T = np.linalg.solve(
            P_yy,
            H @ P_ww
        )
        
        # Update
        self.W = self.W + (y - X).T @ K_T
    
    
    
    def ensample_update(self, x: npt.ArrayLike, y: npt.ArrayLike, var: float):
        
        # basis functions
        H = self.H(np.asarray(x))
        
        # ensamble function value at respective point
        X = np.sum(H * self.W, axis=1)
        
        # Kalman gain P_xy/P_yy
        P_ww = self.cov
        P_yy = np.einsum('kn,nk->k',H ,np.einsum('nm,km->nk', P_ww, H)) + var + self.error_cov
        K_T = (H @ P_ww).T / P_yy
        
        # Update
        self.W = self.W + (K_T * (y.flatten()-X).T).T
        


class BasisFunctionExpansion(abc.ABC):
    
    def __init__(self, n_basis, batch_size=1, jitter_val=1e-8) -> None:
        super().__init__()
        
        self.jitter_val = jitter_val
        
        if not isinstance(n_basis, int):
            raise ValueError(f'Number of basis functions must be an integer but got {type(n_basis)}')
        self.n_basis = n_basis
        
        if batch_size < 1 or not isinstance(batch_size, int):
            raise ValueError(f'Given batch_size must be larger 1 and of type integer')
        self._batch_size = batch_size
        
        self._psi = None
        self._jacobian = None
        self._jacvp = None
        
        self._mean = None
        self._cov = None
        
    
    
    def register_basis_function(self, fcn):
        
        self._psi = jax.jit(fcn)
        self._jacobian = jax.jit(jax.jacfwd(fcn))
        self._jacvp = jax.jit(functools.partial(jax.jvp, fun=fcn))
    
    
    
    def initialize_prior(self, mean, cov):
        
        m = np.asarray(mean).flatten()
        if len(m) != self.n_basis:
            raise ValueError(f'Length of prior mean does not match the number of specified basis functions. Expected {self.n_basis} but got {len(m)}')
        
        c = np.asarray(cov)
        if c.shape[0] != c.shape[1] or len(c.shape) != 2:
            raise ValueError('The prior covariance matrix must be a 2d square matrix')
        if c.shape[0] != self.n_basis:
            raise ValueError('Size of prior covariance matrix must match the number of basis functions')
        
        self._mean = jnp.array(mean).flatten()
        self._cov = jnp.array(cov)
        
        #broadcast to batch
        if self._batch_size > 1:
            self._mean = jnp.broadcast_to(self._mean, (self._batch_size, *self._mean.shape))
            self._cov = jnp.broadcast_to(self._cov, (self._batch_size, *self._cov.shape))
    
    
    
    def psi(self, x):
        
        if self._psi == None:
            raise ValueError('No basis function was registered during init')
        
        return self._psi(jnp.asarray(x))
    
    
    
    def jacobian(self, x):
        
        if self._jacobian == None:
            raise ValueError('No basis function was registered during init')
        
        return self._jacobian(jnp.asarray(x))
    
    
    
    def jacvp(self, x, v):
        
        if self._jacvp == None:
            raise ValueError('No basis function was registered during init')
        
        return self._jacvp(jnp.asarray(x), jnp.asarray(v))
    
    
    
    def __call__(self, x):
        
        if self._mean == None:
            raise ValueError('No prior was initialized')
        
        X = jnp.asarray(x)
        if X.ndim == 1:
            X = X[:,jnp.newaxis]
        
        # evaluate basis functions for input
        Psi = jax.jit(jax.vmap(self.psi))(X)
        Psi = jnp.broadcast_to(Psi, (self._batch_size, *Psi.shape))
        
        # mean value
        f_mean = jnp.einsum('...ij,...j->...i',Psi , self._mean)
        
        # covariance
        f_cov = jnp.einsum('...jn,...ni->...ji', Psi, jnp.einsum('...nj,...ij->...ni', self._cov, Psi))
        
        return jnp.squeeze(f_mean), jnp.squeeze(f_cov)
    
    
    
    def update(self, x: npt.ArrayLike, y: npt.ArrayLike, sigma: npt.ArrayLike):
        
        if self._mean == None:
            raise ValueError('No prior was initialized')
        
        # evaluate features
        psi = self.psi(x)
        
        mean_new, P_new = update_normal_prior(
            self._mean, 
            self._cov, 
            psi, 
            y, 
            sigma,
            self.jitter_val
            )
        
        self._mean = mean_new
        self._cov = P_new



class MultivariateBayesianRegression(abc.ABC):
    
    def __init__(self, n_basis, n_tasks, jitter_val=1e-8) -> None:
        super().__init__()
        
        self.jitter_val = jitter_val
        
        if not isinstance(n_basis, int):
            raise ValueError(f'Number of basis functions must be an integer but got {type(n_basis)}')
        self.n_basis = n_basis
        
        if not isinstance(n_tasks, int):
            raise ValueError(f'Number of basis functions must be an integer but got {type(n_tasks)}')
        self.n_tasks = n_tasks
        
        self._psi = None
        self._jacobian = None
        self._jacvp = None
        
    
    
    def register_basis_function(self, fcn):
        
        self._psi = jax.jit(fcn)
        self._jacobian = jax.jit(jax.jacfwd(fcn))
        self._jacvp = jax.jit(functools.partial(jax.jvp, fun=fcn))
        
        
        
    def psi(self, x):
        
        if self._psi == None:
            raise ValueError('No basis function was registered during init')
        
        return self._psi(jnp.asarray(x))
    
    
    
    def jacobian(self, x):
        
        if self._jacobian == None:
            raise ValueError('No basis function was registered during init')
        
        return self._jacobian(jnp.asarray(x))
    
    
    
    def jacvp(self, x, v):
        
        if self._jacvp == None:
            raise ValueError('No basis function was registered during init')
        
        return self._jacvp(jnp.asarray(x), jnp.asarray(v))



class LinearBayesianRegression(abc.ABC):
    
    def __init__(self, n_basis, jitter_val=1e-8, forgetting_factor=0.001) -> None:
        super().__init__()
        
        self.jitter_val = jitter_val
        
        if not isinstance(n_basis, int):
            raise ValueError(f'Number of basis functions must be an integer but got {type(n_basis)}')
        self.n_basis = n_basis
        
        if forgetting_factor > 1 or forgetting_factor <= 0:
            raise ValueError(f'Given forgetting_factor must be in the interval (0, 1]')
        self._forgetting_factor = forgetting_factor
        
        self._psi = None
        self._jacobian = None
        self._jacvp = None
        
        self._Lambda_inv = None
        self._a = None
        self._b = None
        
        self._RNGKey = jax.random.PRNGKey(np.random.randint(0, 1e8))
    
    
    
    def sample_theta(self, N=1):
        
        self._RNGKey, sample_Key = jax.random.split(self._RNGKey)
        Theta = sample_nig(self._mean, self._Lambda_inv, self._a, self._b, sample_Key, N)
        
        return Theta
    
    
    
    def sample_f(self, x, N=1):
        
        Theta = self.sample_theta(N)
        
        X = jnp.asarray(x)
        if X.ndim == 1:
            X = X[:,jnp.newaxis]
        
        # evaluate basis functions for input
        Psi = jax.jit(jax.vmap(self.psi))(X)
        
        Psi, Theta = jnp.broadcast_arrays(Psi, Theta[...,jnp.newaxis,:])
        
        return jnp.einsum('...j,...j->...', Psi, Theta)
            
    
    
    def register_basis_function(self, fcn):
        
        self._psi = jax.jit(fcn)
        self._jacobian = jax.jit(jax.jacfwd(fcn))
        self._jacvp = jax.jit(functools.partial(jax.jvp, fun=fcn))
    
    
    
    def initialize_prior(self, mean, Lambda, a=1, b=1):
        
        m = np.asarray(mean).flatten()
        if len(m) != self.n_basis:
            raise ValueError(f'Length of prior mean does not match the number of specified basis functions. Expected {self.n_basis} but got {len(m)}')
        
        c = np.asarray(Lambda)
        if c.shape[0] != c.shape[1] or len(c.shape) != 2:
            raise ValueError('The prior covariance matrix must be a 2d square matrix')
        if c.shape[0] != self.n_basis:
            raise ValueError('Size of prior covariance matrix must match the number of basis functions')
        
        self._mean = jnp.array(m)
        self._Lambda = jnp.array(Lambda)
        self._Lambda_inv = jnp.linalg.inv(Lambda)
        self._a = jnp.array(a)
        self._b = jnp.array(b)
        
        
    
    def psi(self, x):
        
        if self._psi == None:
            raise ValueError('No basis function was registered during init')
        
        return self._psi(jnp.asarray(x))
    
    
    
    def jacobian(self, x):
        
        if self._jacobian == None:
            raise ValueError('No basis function was registered during init')
        
        return self._jacobian(jnp.asarray(x))
    
    
    
    def jacvp(self, x, v):
        
        if self._jacvp == None:
            raise ValueError('No basis function was registered during init')
        
        return self._jacvp(jnp.asarray(x), jnp.asarray(v))
    
    
    
    def update(self, x, y):
        
        if self._mean == None:
            raise ValueError('No prior was initialized')
        
        # evaluate features
        psi = self.psi(x)
        
        # forgetting factor
        # self._mean *= 0.99
        self._Lambda *= 1 - self._forgetting_factor
        self._Lambda_inv /= 1 - self._forgetting_factor
        self._a *= 1 - self._forgetting_factor
        self._b *= 1 - self._forgetting_factor
        
        # batched parameter updates
        mu_new, Lambda_new, Lambda_inv_new, a_new, b_new = update_nig_prior(
            self._mean, 
            self._Lambda,
            self._Lambda_inv, 
            self._a, 
            self._b, 
            psi, 
            y
            )
        
        if jnp.linalg.norm(Lambda_new@Lambda_inv_new-jnp.eye(Lambda_new.shape[-1])) > 1e-3:
            Lambda_inv_new = jnp.linalg.inv(Lambda_new)
        
        # assign new parameters
        self._mean = mu_new
        self._Lambda = Lambda_new
        self._Lambda_inv = Lambda_inv_new
        self._a = a_new
        self._b = b_new
    
    
    
    def __call__(self, x):
        
        if self._mean == None:
            raise ValueError('No prior was initialized')
        
        X = jnp.asarray(x)
        if X.ndim == 1:
            X = X[:,jnp.newaxis]
        
        # evaluate basis functions for input
        Psi = jax.jit(jax.vmap(self.psi))(X)
        
        # mean value
        f_mean = jnp.einsum('kj,j->k',Psi , self._mean)
        
        return f_mean