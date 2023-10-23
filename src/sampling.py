import numpy as np
import numpy.typing as npt




def sigma_point_transform(mean, cov, *, alpha: float = 1e-3, beta: float = 2, kappa: float = 0):
    
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
    sigmas[1:] = mean + np.concatenate((U.T,-U.T))
    
    return Wc, Wm, sigmas



def condition_gaussian(
    mean: npt.ArrayLike, 
    cov: npt.ArrayLike, 
    dim, 
    value: npt.ArrayLike
    ):
    
    # mask for all dimensions that aren't in dim
    mask = np.ones(mean.shape[-1], dtype=bool)
    mask[dim] = 0
    
    # Kalman gain
    K = np.linalg.solve(cov[np.ix_(dim,dim)].T, cov[np.ix_(mask,dim)].T).T
    
    # clac conditional gaussian
    new_mean = mean[mask] + K@(value-mean[dim])
    new_cov = cov[np.ix_(mask,mask)] - K@cov[np.ix_(mask,dim)].T
    
    return new_mean, new_cov



def systematic_SISR(w: npt.ArrayLike):
    
    # number of samples
    N = len(w)
    
    # initialize array of indices
    indices = np.zeros((N,), dtype=np.int64)
    
    # select deterministic samples
    U = (np.random.rand() + np.array(range(0, N))) / N
    W = np.cumsum(w, 0)
    
    i, j = 0, 0
    while i < N:
        if U[i] < W[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
        if j >= N:
            if i > N: indices[i:] = j-1
            break
    
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
    dx = (x - y).unsqueeze(-1)
    # d = torch.sum(dx, dim=1)
    return np.exp(-0.5 * dx.transpose(-2,-1) @ np.linalg.solve(cov, dx)).flatten()