import numpy as np
import numpy.typing as npt
import typing as tp
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