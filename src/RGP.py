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
        


class LimitedSupportRBF(BaseBasisFunction):
    
    def __init__(
        self, 
        centers: npt.ArrayLike, 
        lengthscale: npt.ArrayLike,
        support_radius: float = 2
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
        
        if support_radius <= 0:
            raise ValueError("Radius must be positive")
        self.support = support_radius
    
    
    
    @property
    def n_input(self):
        return self._n_input
    
    
    
    @property
    def n_basis(self):
        return self._n_basis
    
    
    
    def __call__(self, x: npt.ArrayLike) -> scipy.sparse.sparray:
        
        x_in = np.asarray(x)
        
        r = scipy.spatial.distance.cdist(
            x_in/self.lengthscale, 
            self.centers/self.lengthscale, 
            'euclidean'
            )
        
        # has_support = scipy.sparse.csc_matrix(r <= self.support)
        
        return np.exp(-0.5*r**2)#has_support.multiply(np.exp(-0.5*r**2)).tocsc()



################################################################################
# GP Models    
        
        
class ApproximateGP():
    
    def __init__(
        self,
        basis_function: BaseBasisFunction,
        w0: npt.ArrayLike,
        cov0: npt.ArrayLike,
        error_cov: float = 1e-3
        ) -> None:
        
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
    
    
    
    def predict(self, x):
        
        H = self.H(x)
        P = H @ self._cov @ H.T + scipy.sparse.diags(np.ones(x.shape[0])*self.error_cov)
        
        return H @ self._mean, P
    
    
    
    def update(self, x: npt.NDArray, y: npt.NDArray, v: npt.NDArray):
        
        H = self.H(x)
        
        e = y.flatten() - H @ self._mean
        
        y_var = scipy.sparse.diags(v)
        f_var = scipy.sparse.diags(np.ones(x.shape[0])*self.error_cov)
        S = H @ self._cov @ H.T + y_var + f_var
        G = scipy.linalg.solve(
            S, 
            (H @ self._cov),
            assume_a='sym'
            )
        
        self._mean = self._mean + (G.T @ e).flatten()
        self._cov = self._cov - G.T @ S @ G