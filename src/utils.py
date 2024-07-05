######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################
# Implements various utilities

import scipy.stats as stats
import numpy as np
import json
import numpy.random as npr


def state_transition_func(xt, t):
    time_t = t+1
    return 0.5*xt + 25*xt/(1+xt**2) + 8*np.cos(1.2*time_t)


def transfer_func(x):
    return x**2/20


def generate_data(f=state_transition_func, g=transfer_func, q=1, r=1, x0=0, seq_len=100, file_name=None):
    x = np.zeros(seq_len)
    y = np.zeros(seq_len)
    q = np.sqrt(q)
    r = np.sqrt(r)
    x[0] = x0  # deterministic initial state
    y[0] = g(x[0]) + r*np.random.normal(size=1)
    
    for t in range(1, seq_len):
        x[t] = f(x[t-1], t-1) + q*np.random.normal(size=1)
        y[t] = g(x[t]) + r*np.random.normal(size=1)

    # save data in a file
    if file_name is not None:
        data = {'x': x.tolist(), 'y': y.tolist()}

        with open(file_name, 'w+') as fh:
            fh.write(json.dumps(data))

    return x, y


def load_data(file_name="data/test.json"):
    with open(file_name, 'r') as fh:
        result = json.loads(fh.read())
    x = result["x"]
    y = result["y"]
    return x, y


def multinomial_resampling(w, size=0):
    if size == 0:
        n = len(w)
    else:
        n = size

    ls = np.random.choice(len(w), size=n, replace=True, p=w)
    return ls
    # u = npr.rand(n)
    # return np.digitize(u, np.cumsum(w))
    # return np.searchsorted(cumsum, u)


def systematic_resampling(w, size=0):
    # Determine number of elements
    if size == 0:
        n = len(w)
    else:
        n = size  # len(ws)

    # Create one single uniformly distributed number
    u_rand = stats.uniform.rvs() / n

    i = np.arange(1, n + 1)
    u = (i - 1) / n + u_rand

    return np.digitize(u, np.cumsum(w))


def stratified_resampling(w, size=0):
    # Determine number of elements
    if size == 0:
        n = len(w)
    else:
        n = size  # len(ws)

    u = (np.arange(n) + npr.rand(n)) / n

    return np.digitize(u, np.cumsum(w))


def ess(w):
    """
    Computes Effective Sample Size
    :param w: weights
    :return: Effective sample size (ESS)
    """
    return 1/np.sum(w**2)
    # return 1 / np.dot(w,w)
