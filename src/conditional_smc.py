######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################

# Implements Conditional particle filter (CPF) or Conditional SMC (CSMC)
# Input:
#   f,g,q,r   - state parameters
#   y       - measurements
#   x0      - initial state
#   N       - number of particles
#   resamplingMethod - resampling methods:
#     multinomial, stratified and systematic resampling methods are supported

# Output:
#   x_star  - sample from target distribution

import numpy as np
from src.utils import multinomial_resampling, systematic_resampling, stratified_resampling
import sys
from scipy.stats import norm


class CSMC(object):
    def __init__(self, f, g, q, r):
        self.f = f
        self.g = g
        self.q = q
        self.r = r
        self.particles = None
        self.normalised_weights = None
        self.ancestors = None
        self.seq_len = 0
        self.n_particles = 100
        self.log_likelihood = 0.0
        self.resampling_method = multinomial_resampling
        self.ancestor_sampling = False

    def resample_step(self, t):
        new_ancestors = self.resampling_method(self.normalised_weights[:, t])
        self.ancestors[:, t] = new_ancestors.astype(int)

    def propagation_step(self, x_ref_t, t):
        # n_minus_1 = self.n_particles - 1
        xpred = self.f(self.particles[:, t - 1], t - 1)
        # propogation step
        new_ancestors = self.ancestors[:, t - 1]
        self.particles[:, t] = xpred[new_ancestors] + np.sqrt(self.q) * np.random.normal(size=self.n_particles)
        if self.ancestor_sampling:
            # Ancestor sampling
            m = norm.logpdf(xpred[new_ancestors], x_ref_t, np.sqrt(self.q))
            const = max(m)  # Subtract the maximum value for numerical stability
            w_as = np.exp(m - const)
            w_as = np.multiply(w_as, self.normalised_weights[:, t - 1])
            w_as = w_as / sum(w_as)  # Save the normalized weights
            self.ancestors[self.n_particles-1, t] = multinomial_resampling(w_as, 1)[0]

    def weighting_step(self, y_t, t):
        ypred = self.g(self.particles[:, t])
        # logweights = norm.logpdf(ypred, y_t, np.sqrt(self.r))
        logweights = -(1 / (2 * self.r)) * (y_t - ypred) ** 2
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        self.normalised_weights[:, t] = w_p / sum(w_p)  # Save the normalized weights

        # accumulate the log-likelihood
        self.log_likelihood = self.log_likelihood + max_weight + np.log(sum(w_p)) - np.log(self.n_particles)

    def generate_particles(self, y, x0, x_ref):
        # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameter is NULL")

        # initialize
        self.seq_len = len(y)
        self.particles = np.zeros((self.n_particles, self.seq_len))
        self.normalised_weights = np.zeros((self.n_particles, self.seq_len))
        self.ancestors = np.zeros((self.n_particles, self.seq_len), dtype=int)

        # Init state, at t=0
        self.particles[:, 0] = x0  # Deterministic initial condition
        self.particles[self.n_particles-1, 0] = x_ref[0]  # reference trajectory
        # self.ancestors[self.n_particles - 1, :] = self.n_particles-1  # fixed ancestors for the ref trajectory

        # weighting step at t=0
        self.weighting_step(y[0], 0)
        # self.ancestors[:, 0] = np.arange(self.n_particles)

        for t in range(1, self.seq_len):
            # resampling step
            self.resample_step(t-1)
            self.ancestors[self.n_particles - 1, t-1] = self.n_particles - 1

            # propagation step
            self.propagation_step(x_ref[t], t)
            self.particles[self.n_particles - 1, t] = x_ref[t]

            # weighting step
            self.weighting_step(y[t], t)

        self.ancestors[:, self.seq_len-1] = np.arange(self.n_particles)
        self.ancestors = self.ancestors.astype(int)

        # sample a state trajectory from _particles

    def get_state_trajectory(self):
        if self.particles is None:
            print("call generateWeightedParticles first")
            exit(0)

        x_star = np.zeros(self.seq_len)
        indx = multinomial_resampling(self.normalised_weights[:, self.seq_len - 1], 1)[0]

        for t in reversed(range(self.seq_len)):
            indx = self.ancestors[indx, t]
            x_star[t] = self.particles[indx, t]
        return x_star

    # Given theta estimate states using PG
    def sample_states(self, y, x0, x_ref, n_particles=100, ancestor_sampling=False,
                      resampling_method='multi', max_iter=100):

        # set resampling method
        if resampling_method == 'systematic':
            self.resampling_method = systematic_resampling
        elif resampling_method == 'stratified':
            self.resampling_method = stratified_resampling
        else:
            self.resampling_method = multinomial_resampling

        self.ancestor_sampling = ancestor_sampling
        self.n_particles = n_particles

        x_new_ref = x_ref.copy()

        for i in range(max_iter):
            self.generate_particles(y, x0, x_new_ref)
            x_new_ref = self.get_state_trajectory()

        return x_new_ref

    def set_noise_param(self, q, r):
        self.q = q
        self.r = r

        # returns log likelihood

    def get_log_likelihood(self):
        return self.log_likelihood


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import state_transition_func, transfer_func, generate_data
    # Set up some parameters
    n = 100  # Number of particles
    T = 100   # Length of data record
    f1 = state_transition_func
    g1 = transfer_func
    r1 = .1
    q1 = 1.
    xref = np.zeros(T)
    x_0 = np.zeros(n)
    # Generate data
    # np.random.seed(123)
    data_x, data_y = generate_data(r=r1, q=q1, seq_len=T)

    cpf = CSMC(f1, g1, q=q1, r=r1)
    x = cpf.sample_states(data_y, x0=0, x_ref=xref, n_particles=n, resampling_method='multi', max_iter=100)
    plt.figure(figsize=(10, 4))
    print(max(np.abs(x - data_x)))
    plt.plot(x - data_x, color='indigo')
    plt.xlabel("time")
    plt.ylim((-20, 20))
    plt.ylabel("error in state estimate")
    plt.savefig("plots/err_x_BPF")


    plt.figure(figsize=(10, 4))
    plt.plot(data_x, marker='o', label='Real States', markersize=3)
    plt.plot(x, label='Conditional SMC')
    plt.legend()

    plt.show()