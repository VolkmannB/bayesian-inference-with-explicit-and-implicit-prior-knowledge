import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import scipy.linalg
import abc



def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
  
  n = A.shape[0]
  LU = scipy.sparse.linalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition
  
  if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
    return LU.L.dot( scipy.sparse.diags(LU.U.diagonal()**0.5) )
  else:
    raise ValueError('The matrix is not positive definite')



def sq_dist(x1: npt.ArrayLike, x2: npt.ArrayLike) -> npt.NDArray:
    """
    This function calculates the squared eicledian distance ||x||_2^2 between 
    all pairs of vectors in x1 and x2.

    Args:
        x1 (npt.ArrayLike): An [..., n_a, m] array of coordinates.
        x2 (npt.ArrayLike): an [..., n_b, m] array of coordinates.

    Returns:
        npt.NDArray: An [..., n_a, n_b] array containing the squared euclidean distance 
        between each pair of vectors.
    """
    
    a = np.asarray(x1)
    b = np.asarray(x2)
    
    distance = np.sum(
            (a[..., np.newaxis,:] - b[...,np.newaxis,:,:])**2,
            axis=-1
            )
    
    return distance
    



################################################################################
# Distribution

class BaseDistribution(abc.ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def sample():
        raise NotImplementedError()



class SparseMultivariateNormal(BaseDistribution):
    
    def __init__(self, mean, cov) -> None:
        super().__init__()

        # check sizes 
        if cov.shape[0] != cov.shape[1] or len(cov.shape) != 2:
            raise ValueError('Covariance matrix must be quadratic')
        if mean.size != cov.shape[0]:
            raise ValueError('Size of mean vector does not match covariance matrix')
        
        self._mean = np.array(mean)
        self._chol_cov = sparse_cholesky(
            scipy.sparse.csc_matrix(np.asarray(cov))
            )
    
    
    
    def sample(self, N: int):
        
        return self._mean + np.random.randn(N, self._mean.size) @ self._chol_cov.T
    
    
    
    @property
    def cov(self):
        return self._chol_cov @ self._chol_cov.T
    
    
    
    @property
    def chol_cov(self):
        return self._chol_cov
    
    
    
    @property
    def mean(self):
        return self._mean
    
    
    
class MultivariateNormal(BaseDistribution):
    
    def __init__(self, mean, cov) -> None:
        super().__init__()

        # check sizes 
        if cov.shape[0] != cov.shape[1] or len(cov.shape) != 2:
            raise ValueError('Covariance matrix must be quadratic')
        if mean.size != cov.shape[0]:
            raise ValueError('Size of mean vector does not match covariance matrix')
        
        self._mean = np.array(mean)
        self._chol_cov = np.linalg.cholesky(cov)
    
    
    
    def sample(self, N: int):
        
        return self._mean + np.random.randn(N, self._mean.size) @ self._chol_cov.T
    
    
    
    @property
    def cov(self):
        return self._chol_cov @ self._chol_cov.T
    
    
    
    @property
    def chol_cov(self):
        return self._chol_cov
    
    
    
    @property
    def mean(self):
        return self._mean
        
        



################################################################################
# Basis Function

class BaseBasisFunction(abc.ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()
    
    @property
    @abc.abstractmethod
    def n_basis(self):
        pass
    
    @property
    @abc.abstractmethod
    def n_input(self):
        pass
        


class GaussianRBF(BaseBasisFunction):
    
    def __init__(
        self, 
        centers: npt.ArrayLike, 
        lengthscale: npt.ArrayLike
        ) -> None:
        
        super().__init__()
        
        self.centers = np.array(centers)
        
        self._n_input = centers.shape[1]
        self._n_basis = centers.shape[0]
        
        if np.any(lengthscale<=0):
            raise ValueError("Lengthscale must be positive")
        if len(lengthscale) != 1 and len(lengthscale) != self.n_input:
            raise ValueError("Lengthscale must be compatible with input dimensions or be a scalar")
        self.lengthscale = np.array(lengthscale).flatten()
    
    
    
    @property
    def n_input(self):
        return self._n_input
    
    
    
    @property
    def n_basis(self):
        return self._n_basis
    
    
    
    def __call__(self, x: npt.ArrayLike) -> scipy.sparse.sparray:
        
        x_in = np.asarray(x)
        
        r = sq_dist(
            x_in/self.lengthscale, 
            self.centers/self.lengthscale
            )
        
        return np.exp(-0.5*r)



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
        basis_function: BaseBasisFunction,
        w0: npt.ArrayLike,
        cov0: npt.ArrayLike,
        error_cov: float = 1e-3
        ) -> None:
        
        super().__init__()
        
        if not isinstance(basis_function, BaseBasisFunction):
            raise ValueError("basis_function must be a BaseBasisFunction object")
        self.H = basis_function
        
        if len(w0) != basis_function.n_basis:
            raise ValueError("Number of weights does not match basis functions. Expected {0} but got {1}".format(basis_function.n_basis, len(w0))) 
        if cov0.shape[0] != cov0.shape[1] or len(cov0.shape) != 2:
            raise ValueError('Covariance matrix must be quadratic')
        if w0.size != cov0.shape[0]:
            raise ValueError('Size of mean vector does not match covariance matrix')
        self._mean = np.array(w0)
        self._cov = cov0
        
        if error_cov <= 0:
            raise ValueError("Error variance must be positive")
        self.error_cov = error_cov
    
    
    
    @property
    def mean(self):
        return self._mean
    
    
    
    @property
    def cov(self):
        return self._cov
    
    
    
    def predict(self, x):
        
        # basis functions
        H = self.H(x)
        
        # covariance matrix / einsum for broadcasting cov @ H.T
        P = H @ np.einsum('nm,...km->...nk', self._cov, H) + np.diag(np.ones(x.shape[-2])*self.error_cov)
        
        return H @ self._mean, P
    
    
    
    def update(self, x: npt.NDArray, y: npt.NDArray, v: npt.NDArray):
        
        if len(x.shape[:-2]) != 0:
            x = x.reshape((-1,self.H.n_input))
        
        # basis function
        H = self.H(x)
        
        # residuum
        e = y.flatten() - H @ self._mean
        
        # model and measurment uncertainty
        y_var = np.diag(v)
        f_var = np.diag(np.ones(x.shape[-2])*self.error_cov)
        
        # covariance matrix / einsum for broadcasting cov @ H.T
        S = H @ np.einsum('nm,km->nk', self._cov, H) + y_var + f_var
        
        # gain matrix
        G = np.linalg.solve(
            S, 
            (H @ self._cov)
            )
        
        self._mean = self._mean + (np.swapaxes(G, -2, -1) @ e).flatten()
        self._cov = self._cov - np.swapaxes(G, -2, -1) @ S @ G



class EnsambleGP(BaseGP):
    
    def __init__(
        self,
        basis_function: BaseBasisFunction,
        w0: npt.ArrayLike,
        cov0: npt.ArrayLike,
        N: int,
        error_cov: float = 1e-3
        ) -> None:
        
        super().__init__()
        
        if not isinstance(basis_function, BaseBasisFunction):
            raise ValueError("basis_function must be a BaseBasisFunction object")
        self.H = basis_function
        
        if len(w0) != basis_function.n_basis:
            raise ValueError("Number of weights does not match basis functions. Expected {0} but got {1}".format(basis_function.n_basis, len(w0))) 
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
        
    
    
    @property
    def mean(self):
        return self.W.mean(axis=0)
    
    
    
    @property
    def cov(self):
        return np.cov(self.W.T)
    
    
    
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
        P_ww = np.cov(self.W.T)
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
        P_ww = np.cov(self.W.T)
        P_yy = np.einsum('kn,nk->k',H ,np.einsum('nm,km->nk', P_ww, H)) + var + self.error_cov
        K_T = (H @ P_ww).T / P_yy
        
        # Update
        self.W = self.W + (K_T * (y.flatten()-X).T).T