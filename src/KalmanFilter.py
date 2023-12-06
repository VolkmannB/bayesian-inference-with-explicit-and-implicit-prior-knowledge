import numpy as np
import numpy.typing as npt
import typing as tp
import itertools
import abc



def sigma_point_transform(
    mean, 
    cov, 
    *, 
    alpha: float = 1e-3, 
    beta: float = 2, 
    kappa: float = 0):
    
    n = len(mean)
    
    sigmas = np.zeros((2*n+1,n))
    Wc = np.zeros((2*n+1,1))
    Wm = np.zeros((2*n+1,1))
    
    lambda_ = alpha**2 * (n + kappa) - n
    
    # calculate weights
    c = .5 / (n + lambda_)
    Wc = np.full(2*n + 1, c)
    Wm = np.full(2*n + 1, c)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    Wm[0] = lambda_ / (n + lambda_)
    
    # calculate sigma points
    U = np.linalg.cholesky((lambda_ + n)*cov)
    sigmas[0] = mean
    sigmas[1:] = mean + np.concatenate((U,-U))
    
    return Wc, Wm, sigmas



def GaussHermiteCubature(
    mean,
    cov,
    N
    ):
    
    # generate one dimensional points
    x, w = np.polynomial.hermite.hermgauss(N)
    
    # generate cartesian product for n dimensional case
    ind = np.array(list(itertools.combinations_with_replacement(x, mean.size)), dtype=int)
    
    # transform points to distribution
    w = np.prod(w[ind], axis=1)
    X = mean + x[ind] @ np.linalg.cholesky(cov).T
    
    return w, X



def systematic_SISR(w: npt.ArrayLike):
    
    # number of samples
    N = len(w)
    
    # initialize array of indices
    # indices = np.zeros((N,), dtype=np.int64)
    
    # select deterministic samples
    U = (np.random.rand() + np.array(range(0, N))) / N
    W = np.cumsum(w, 0)
    
    i, j = 0, 0
    # while i < N:
    #     if U[i] < W[j]:
    #         indices[i] = j
    #         i += 1
    #     else:
    #         j += 1
    #     if j >= N:
    #         if i > N: indices[i:] = j-1
    #         break
    indices = np.searchsorted(W, U)
    
    return indices



def squared_error(x, y, cov):
    """
    RBF kernel, supporting masked values in the observation
    Parameters:
    -----------
    x : array (N,D) array of values
    y : array (N,D) array of values

    Returns:
    -------

    distance : scalar
        Total similarity, using equation:

            d(x,y) = e^((-1 * (x - y) ** 2) / (2 * sigma ** 2))

        summed over all samples. Supports masked arrays.
    """
    dx = x - y
    
    t = np.linalg.solve(cov, dx.T).T
    r = np.einsum('nd,nd->n',dx, t)
    
    return np.exp(-0.5 * r)



class BaseFilter(abc.ABC):
    
    def __init__(
        self,
        Q: npt.ArrayLike, 
        R: npt.ArrayLike
        ) -> None:
        
        super().__init__()
        
        # check sizes
        Q_ = np.array(Q)
        R_ = np.array(R)
        if R_.shape[0] != R_.shape[1]:
            raise ValueError('Measurment covariance is not square')
        if Q_.shape[0] != Q_.shape[1]:
            raise ValueError('Process covariance is not square')
        self.Q = Q_
        self.R = R_
        
        self._n_x = Q_.shape[0]
        self._n_y = R_.shape[0]
    
    @abc.abstractmethod
    def predict(self):
        pass
    
    @abc.abstractmethod
    def update(self):
        pass
    
    @property
    @abc.abstractmethod
    def x(self):
        pass
    
    @property
    @abc.abstractmethod
    def P(self):
        pass



class UnscentedKalmanFilter(BaseFilter):
    
    def __init__(
        self, 
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        f_x: tp.Callable, 
        f_y: tp.Callable,
        Q: npt.ArrayLike, 
        R: npt.ArrayLike,
        alpha: float = 1e-3, 
        beta: float = 2, 
        kappa: float = 0
        ) -> None:
        
        super().__init__(Q, R)
        
        # set state and measurment model
        if not callable(f_x):
            raise ValueError('State space model must be callable')
        self.f_x = f_x
        if not callable(f_y):
            raise ValueError('Measurment model must be callable')
        self.f_y = f_y
        
        # set state prior
        x_ = np.array(x0)
        P_ = np.array(P0)
        if x_.shape[0] != self._n_x:
            raise ValueError('Prior mean for state does not match noise covariance matrix')
        if P_.shape[0] != self._n_x or P_.shape[1] != self._n_x:
            raise ValueError('Shape of prior covariance for state does not match noise covariance matrix')
        self._x = x_
        self._P = P_
        
        # these will always be a copy of x,P after predict() is called
        self.x_prior = self._x.copy()
        self.P_prior = self._P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self._x.copy()
        self.P_post = self._P.copy()
        
        # sigma point placeholders
        self.sigma_x = np.zeros((2*self._n_x+1, self._n_x))
        # self.sigma_y = np.zeros((2*self._n_x+1, self._n_y))
        self.Wm = np.zeros((2*self._n_x+1,))
        self.Wc = np.zeros((2*self._n_x+1,))
        
        # parameters for unscented transformation
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
    
    
    
    def predict(self, **fx_args):
        
        # save prior
        self.x_prior = np.copy(self._x)
        self.P_prior = np.copy(self._P)
        
        # create sigma points
        Wc, Wm, sigma_x = sigma_point_transform(
            self._x, 
            self._P,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa
            )
        self.Wm = Wm
        self.Wc = Wc
        
        # transform sigma points
        sigma_x = self.f_x(sigma_x, **fx_args)
        self.sigma_x = sigma_x
        
        # calculate predictive distribution
        self._x = sigma_x.T @ Wm
        sigma_x = sigma_x - self._x
        P_ = np.einsum('mn, mN->mnN', sigma_x, sigma_x)
        self._P = np.einsum('m, mnN->nN', Wc, P_) + self.Q
        
        
        
    def update(self, y:npt.ArrayLike, **fy_args):
        
        # get sigma points for measurment
        sigma_y = self.f_y(self.sigma_x, **fy_args)
        
        # get measurment distribution
        y_ = sigma_y.T @ self.Wm
        sigma_y = sigma_y - y_
        P_yy_ = np.einsum('mn, mN->mnN', sigma_y, sigma_y)
        P_yy = np.einsum('m, mnN->nN', self.Wc, P_yy_) + self.R
        
        # get Kalman gain
        P_xy_ = np.einsum('mn, mN->mnN', self.sigma_x-self._x, sigma_y)
        P_xy = np.einsum('m, mnN->nN', self.Wc, P_xy_)
        K_T = np.linalg.solve(P_yy,P_xy.T)
        
        # calculate measurment update
        self._x = self._x + K_T.T @ (y - y_)
        self._P = self._P - K_T.T @ P_yy @ K_T
        
        # save posterior
        self.x_post = np.copy(self._x)
        self.P_post = np.copy(self._P)
    
    @property
    def x(self):
        return self._x
    
    @property
    def P(self):
        return self._P



class ExtendedKalmanFilter(BaseFilter):
    
    def __init__(
        self, 
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        f_x: tp.Callable, 
        F: tp.Callable, 
        f_y: tp.Callable,
        H: tp.Callable, 
        Q: npt.ArrayLike, 
        R: npt.ArrayLike
        ) -> None:
        
        super().__init__(Q, R)
        
        # set state and measurment model
        if not callable(f_x) or not callable(F):
            raise ValueError('State space model must be callable')
        self.f_x = f_x
        self.F = F
        if not callable(f_y) or not callable(H):
            raise ValueError('Measurment model must be callable')
        self.f_y = f_y
        self.H = H
        
        # set state prior
        x_ = np.array(x0)
        P_ = np.array(P0)
        if x_.shape[0] != self._n_x:
            raise ValueError('Prior mean for state does not match noise covariance matrix')
        if P_.shape[0] != self._n_x or P_.shape[1] != self._n_x:
            raise ValueError('Shape of prior covariance for state does not match noise covariance matrix')
        self._x = x_
        self._P = P_
        
        # these will always be a copy of x,P after predict() is called
        self.x_prior = self._x.copy()
        self.P_prior = self._P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self._x.copy()
        self.P_post = self._P.copy()
    
    @property
    def x(self):
        return self._x
    
    @property
    def P(self):
        return self._P
    
    
    
    def predict(self, **fx_args):
        
        # save prior
        self.x_prior = np.copy(self._x)
        self.P_prior = np.copy(self._P)
        
        # prediction
        self._x = self.f_x(self._x, **fx_args)
        F = self.F(self._x, **fx_args)
        self._P = F @ self._P @ F.T + self.Q
    
    
    
    def update(self, y, **fy_args):
        
        y_ = y - self.f_y(self.x_, **fy_args)
        H = self.H(self.x_, **fy_args)
        S = H @ self._P @ H.T + self.R
        
        # Kalman gain
        K_T = np.linalg.solve(S, H @ self._P)
        
        # update
        self._x = self._x + K_T.T @ y_
        self._P = self._P - K_T.T @ H @ self._P
        
        # save posterior
        self.x_post = np.copy(self._x)
        self.P_post = np.copy(self._P)
        


class EnsambleKalmanFilter(BaseFilter):
    
    def __init__(
        self, 
        N: int, 
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        f_x: tp.Callable, 
        f_y: tp.Callable,
        Q: npt.ArrayLike, 
        R: npt.ArrayLike
        ) -> None:
        
        super().__init__(Q, R)
        
        # set state and measurment model
        if not callable(f_x):
            raise ValueError('State space model must be callable')
        self.f_x = f_x
        if not callable(f_y):
            raise ValueError('Measurment model must be callable')
        self.f_y = f_y
        
        # generate ensamble from prior
        if not isinstance(N, int):
            raise ValueError('Ensamble size must be an integer')
        x_ = np.array(x0)
        P_ = np.array(P0)
        if x_.shape[0] != self._n_x:
            raise ValueError('Prior mean for state does not match noise covariance matrix')
        if P_.shape[0] != self._n_x or P_.shape[1] != self._n_x:
            raise ValueError('Shape of prior covariance for state does not match noise covariance matrix')
        self._sigma_x = x_ + np.random.randn(N,x_.size) @ np.linalg.cholesky(P_).T
        
        # these will always be a copy of x,P after predict() is called
        self._sigma_x_prior = self._sigma_x.copy()

        # these will always be a copy of x,P after update() is called
        self._sigma_x_post = self._sigma_x.copy()
    
    
    
    @property
    def x_prior(self):
        return self._sigma_x_prior.mean(axis=0)
    
    
    
    @property
    def P_prior(self):
        return np.cov(self._sigma_x_prior.T)
    
    
    
    @property
    def x_post(self):
        return self._sigma_x_post.mean(axis=0)
    
    
    
    @property
    def P_post(self):
        return np.cov(self._sigma_x_post.T)
    
    
    
    @property
    def x(self):
        return self._sigma_x.mean(axis=0)
    
    
    
    @property
    def P(self):
        return np.cov(self._sigma_x.T)
    
    
    
    def predict(self, **fx_args):
        
        # save prior
        self._sigma_x_prior = self._sigma_x
        
        # forward simulation
        w = np.random.randn(*self._sigma_x.shape) @ np.linalg.cholesky(self.Q).T
        self._sigma_x = self.f_x(self._sigma_x, **fx_args) + w
        
        
    
    def update(self, y, **fy_args):
        
        # ensamble mean
        x_ = self._sigma_x.mean(axis=0)
        
        # simulate measurments
        # v = np.random.randn(self._sigma_x.shape[0], self.R.shape[0]) @ np.linalg.cholesky(self.R).T
        sigma_y = self.f_y(self._sigma_x, **fy_args) #+ v
        y_ = self.f_y(x_[None,:], **fy_args)
        
        # centered measurement ensamble
        Y_ = sigma_y-y_
        
        # calculate joint
        P_xy = 1/(self._sigma_x.shape[0]-1) * np.einsum('ji,jk->ik', self._sigma_x-x_, Y_)
        P_yy = 1/(self._sigma_x.shape[0]-1) * np.einsum('ji,jk->ik', Y_, Y_) + self.R
        # P = np.cov(np.concatenate((self._sigma_x, sigma_y), axis=1).T)
        
        # Kalman gain P_xy/P_yy - due to solve() K^T is actually calculated
        # n_x = self._sigma_x.shape[1]
        # K_T = np.linalg.solve(P[n_x:, n_x:], P[:n_x,n_x:].T)
        K_T = np.linalg.solve(P_yy, P_xy.T)
        
        # update
        self._sigma_x = self._sigma_x + (y - sigma_y) @ K_T
        
        # save posterior
        self._sigma_x_post = self._sigma_x
    
    
    
    def resample(self):
        
        P = self.P
        
        w = squared_error(self._sigma_x, self.x, P)
        w /= np.sum(w)
        
        ind = systematic_SISR(w)
        
        self._sigma_x = self._sigma_x[ind,:]
        
        return ind



class GaussHermiteKalmanFilter(BaseFilter):
    
    def __init__(
        self, 
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        f_x: tp.Callable, 
        f_y: tp.Callable,
        Q: npt.ArrayLike, 
        R: npt.ArrayLike,
        N: int = 3
        ) -> None:
        
        super().__init__(Q, R)
        
        # set state and measurment model
        if not callable(f_x):
            raise ValueError('State space model must be callable')
        self.f_x = f_x
        if not callable(f_y):
            raise ValueError('Measurment model must be callable')
        self.f_y = f_y
        
        # set state prior
        x_ = np.array(x0)
        P_ = np.array(P0)
        if x_.shape[0] != self._n_x:
            raise ValueError('Prior mean for state does not match noise covariance matrix')
        if P_.shape[0] != self._n_x or P_.shape[1] != self._n_x:
            raise ValueError('Shape of prior covariance for state does not match noise covariance matrix')
        self._x = x_
        self._P = P_
        
        # these will always be a copy of x,P after predict() is called
        self.x_prior = self._x.copy()
        self.P_prior = self._P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self._x.copy()
        self.P_post = self._P.copy()
        
        # order of cubature transformation
        self.N = N
    
    
    
    def predict(self, **fx_args):
        
        # save prior
        self.x_prior = np.copy(self._x)
        self.P_prior = np.copy(self._P)
        
        # create sigma points
        w, sigma_x = GaussHermiteCubature(
            self._x, 
            self._P,
            N=self.N
            )
        
        # transform sigma points
        sigma_x = self.f_x(sigma_x, **fx_args)
        
        # calculate predictive distribution
        self._x = sigma_x.T @ w
        sigma_x = sigma_x - self._x
        P_ = np.einsum('mn, mN->mnN', sigma_x, sigma_x)
        self._P = np.einsum('m, mnN->nN', w, P_) + self.Q
        
        
        
    def update(self, y:npt.ArrayLike, **fy_args):
        
        # generate sigma points
        w, sigma_X = GaussHermiteCubature(
            self._x, 
            self._P,
            N=self.N
            )
        
        # get sigma points for measurment
        sigma_y = self.f_y(sigma_X, **fy_args)
        
        # get measurment distribution
        y_ = sigma_y.T @ w
        sigma_y = sigma_y - y_
        P_yy_ = np.einsum('mn, mN->mnN', sigma_y, sigma_y)
        P_yy = np.einsum('m, mnN->nN', w, P_yy_) + self.R
        
        # get Kalman gain
        P_xy_ = np.einsum('mn, mN->mnN', sigma_X - self._x, sigma_y)
        P_xy = np.einsum('m, mnN->nN', w, P_xy_)
        K_T = np.linalg.solve(P_yy,P_xy.T)
        
        # calculate measurment update
        self._x = self._x + K_T.T @ (y - y_)
        self._P = self._P - K_T.T @ P_yy @ K_T
        
        # save posterior
        self.x_post = np.copy(self._x)
        self.P_post = np.copy(self._P)
    
    @property
    def x(self):
        return self._x
    
    @property
    def P(self):
        return self._P