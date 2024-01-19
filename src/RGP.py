import numpy as np
import numpy.typing as npt
import abc
import typing as tp

import jax.numpy as jnp
import jax



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
    
    return distance
    


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
        

        
        
class ApproximateGP(BaseGP):
    
    def __init__(
        self,
        basis_function: callable,
        n_basis: int,
        batch_shape: npt.ArrayLike = (),
        jitter_val: float = 1e-8
        ) -> None:
        
        super().__init__()
        
        # check for type
        if not callable(basis_function):
            raise ValueError("basis_function must be a callable object")
        self.H = jax.jit(basis_function)
        self.n_H = n_basis
        
        # initialize mean and covariance with default values
        self._mean = np.zeros((*batch_shape, n_basis)) # batch of mean vectors
        self._cov = np.zeros((*batch_shape, n_basis, n_basis)) # batch ov covariance matrices
        idx = np.arange(n_basis)
        self._cov[..., idx, idx] = 1 # set main diagonals to zero
        
        self.batch_shape = batch_shape
        
        # set base model uncertainty
        if jitter_val <= 0:
            raise ValueError("Error variance must be positive")
        self.jitter_val = jitter_val
    
    
    
    @property
    def mean(self):
        return self._mean
    
    
    
    @property
    def cov(self):
        return self._cov
    
    
    
    def predict(self, x: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Performs a prediction for the function at the given points x.

        Args:
            x (npt.ArrayLike): An [...,m,n] array of input points.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: An [...,m] for the mean value of 
            the function and an [...,m,m] matrix for the covariance.
        """
        
        # check shapes
        x_in = np.asarray(x)
        if not np.all(x_in.shape[:-2] == self.batch_shape):
            batch_shape = np.broadcast_shapes(x_in.shape[:-2], self.batch_shape)
            X = np.broadcast_to(x_in, (*batch_shape, *x_in.shape[-2:]))
        else:
            X = np.asarray(x)
        
        # basis functions
        H = self.H(X)
        
        # covariance matrix / einsum for broadcasting cov @ H.T
        P = H @ np.einsum('...nm,...km->...nk', self._cov, H)
        
        # add jitter
        P += np.eye(P.shape[-1])*self.jitter_val
        
        return np.einsum('...nj,...j->...n', H, self._mean), P
    
    
    
    def update(self, x: npt.NDArray, y: npt.NDArray, Q: npt.NDArray):
        """
        Performs the parameter update for the Gaussian process given input 
        points x, observations y and observation covariance Q.

        Args:
            x (npt.NDArray): An [...,m,n] array of input points.
            y (npt.NDArray): An [...,m] array of observations.
            Q (npt.NDArray): An [...,m] matrix of representing the covariance 
            of the observations.
        """
        
        # check shapes
        x_in = np.asarray(x)
        if not np.all(x_in.shape[:-2] == self.batch_shape):
            batch_shape = np.broadcast_shapes(x_in.shape[:-2], self.batch_shape)
            X = np.broadcast_to(x_in, (*batch_shape, *x_in.shape[-2:]))
        else:
            X = np.asarray(x)
        
        # basis function
        H = self.H(X)
        
        # residuum / einsum for batch broadcasting / n input points; j basis functions
        e = y - np.einsum('...nj,...j->...n', H, self._mean)
        
        # covariance matrix / einsum for broadcasting cov @ H.T
        S = H @ np.einsum('...nm,...km->...nk', self._cov, H) + np.diag(Q)
        
        # add jitter
        S += np.eye(S.shape[-1])*self.jitter_val
        
        # gain matrix
        G = np.linalg.solve(
            S, 
            (H @ self._cov)
            )
        
        # k measurments; j basis functions
        self._mean = self._mean + np.einsum('...kj,...k->...j', G, e)
        self._cov = self._cov - np.swapaxes(G, -2, -1) @ S @ G
    
    
    
    def sample(self, x: npt.ArrayLike):
        
        mean, P = self.predict(x)
        
        return mean + np.linalg.cholesky(P) @ np.random.randn(mean.shape[-1])



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