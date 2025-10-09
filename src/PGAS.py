import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import numpy as np
from typing import Callable
from tqdm import tqdm

import src.BayesianInferrence as BI
import src.Filtering as Filtering
from src.Filtering import reconstruct_trajectory


class condSequentialMonteCarlo(eqx.Module):
    N_samples: int
    observations: jax.Array
    inputs: jax.Array
    init_state_mean: jax.Array
    init_state_cov: jax.Array
    likelihood_fcn: Callable[[jax.Array, jax.Array], jax.Array]
    basis_fcn: Callable[[jax.Array, jax.Array], jax.Array]
    dim_basis: float

    def __init__(
        self,
        N_samples: int,
        observations: jax.Array,
        inputs: jax.Array,
        init_state_mean: jax.Array,
        init_state_cov: jax.Array,
        likelihood_fcn: Callable[[jax.Array, jax.Array], jax.Array],
        basis_fcn: Callable[[jax.Array, jax.Array], jax.Array],
    ):
        self.N_samples = N_samples
        self.observations = observations
        self.inputs = inputs
        self.init_state_mean = init_state_mean
        self.init_state_cov = init_state_cov
        self.likelihood_fcn = likelihood_fcn
        self.basis_fcn = basis_fcn
        self.dim_basis = len(
            self.basis_fcn(self.init_state_mean, self.inputs[0])
        )

    def _generate_auxiliary_states(
        self,
        state: jax.Array,
        time: int,
        coeff_mat: jax.Array,
    ) -> tuple[jax.Array]:
        # generate auxiliary state
        basis = jax.vmap(self.basis_fcn, in_axes=(0, None))(
            state, self.inputs[time]
        )
        aux_state = jnp.einsum("kj,ij->ik", coeff_mat, basis)

        return aux_state

    def _draw_states(
        self,
        key: jax.Array,
        time: int,
        state: jax.Array,
        coeff_mat: jax.Array,
        error_cov: jax.Array,
    ) -> jax.Array:
        basis = jax.vmap(self.basis_fcn, in_axes=(0, None))(
            state, self.inputs[time]
        )
        mean = jnp.einsum("ik,jk->ji", coeff_mat, basis)

        int_var = jax.vmap(
            jax.random.multivariate_normal,
            in_axes=(0, 0, None),
        )(jax.random.split(key, self.N_samples), mean, error_cov)

        return int_var

    def step(
        self,
        key: jax.Array,
        time: int,
        log_weights: jax.Array,
        state: jax.Array,
        coeff_mat: jax.Array,
        error_cov: jax.Array,
        ref_state: jax.Array,
    ) -> tuple[jax.Array]:
        # generate auxiliary states
        aux_state = self._generate_auxiliary_states(state, time, coeff_mat)

        # compute first stage weights
        log_likelihood_aux = jax.vmap(
            self.likelihood_fcn,
            in_axes=(None, 0, None),
        )(
            self.observations[time],
            aux_state,
            self.inputs[time],
        )
        log_weights_aux = log_likelihood_aux + log_weights
        aux_weights = jax.nn.softmax(log_weights_aux)

        # draw ancestors
        key, key_ancestor = jax.random.split(key)
        a_indices = Filtering.systematic_SISR(key_ancestor, aux_weights)

        # draw ancestor for reference trajectory
        h_x = jax.vmap(
            jsp.stats.multivariate_normal.logpdf,
            in_axes=(None, 0, None),
        )(
            ref_state,
            aux_state,
            error_cov,
        )
        log_weights_ancestor = log_weights_aux + h_x
        weights_ancestor = jax.nn.softmax(log_weights_ancestor)

        # sample an ancestor index for reference trajectory
        key, key_ancestor = jax.random.split(key)
        ref_idx = jnp.searchsorted(
            jnp.cumsum(weights_ancestor), jax.random.uniform(key_ancestor)
        )

        # set ancestor index
        a_indices = a_indices.at[-1].set(ref_idx)

        # draw new state
        key, key_state = jax.random.split(key)
        new_state = self._draw_states(
            key_state, time, state, coeff_mat, error_cov
        )
        new_state = new_state.at[-1].set(ref_state)

        # compute new weights
        new_log_weights = (
            jax.vmap(
                self.likelihood_fcn,
                in_axes=(None, 0, None),
            )(
                self.observations[time],
                new_state,
                self.inputs[time],
            )
            - log_likelihood_aux[a_indices]
        )

        return (
            new_log_weights,
            new_state,
            a_indices,
        )

    def _init_algorithm(
        self,
        key: jax.Array,
    ) -> tuple[jax.Array]:
        N_steps = self.observations.shape[0]
        state_trace = np.zeros(
            (N_steps, self.N_samples, self.init_state_mean.shape[0])
        )
        log_weights_trace = np.zeros((N_steps, self.N_samples))
        ancestor_trace = np.zeros((N_steps, self.N_samples))

        # sample initial states
        state_trace[0] = jax.random.multivariate_normal(
            key,
            self.init_state_mean,
            self.init_state_cov,
            shape=(self.N_samples,),
        )

        return state_trace, log_weights_trace, ancestor_trace

    def __call__(
        self,
        key: jax.Array,
        ref_state: jax.Array,
        coeff_mat: jax.Array,
        error_cov: jax.Array,
    ) -> tuple[jax.Array]:
        # run initialization
        key, key_init = jax.random.split(key)
        (
            state_trace,
            log_weights_trace,
            ancestor_trace,
        ) = self._init_algorithm(
            key_init,
        )

        # set initial reference trajectory
        state_trace[0, -1] = ref_state[0]

        # run loop
        jit_step = eqx.filter_jit(self.step)
        N_steps = self.observations.shape[0]
        for time in tqdm(
            range(1, N_steps), desc="Processing time steps", leave=False
        ):
            # run a step of the loop in Algorithm 3
            key, key_step = jax.random.split(key)
            (
                new_log_weights,
                new_state,
                a_indices,
            ) = jit_step(
                key_step,
                jnp.array(time),
                log_weights_trace[time - 1],
                state_trace[time - 1],
                coeff_mat,
                error_cov,
                ref_state[time],
            )

            # save results in trace
            state_trace[time] = new_state
            log_weights_trace[time] = new_log_weights
            ancestor_trace[time - 1] = a_indices

        # sample a trajectory
        w = jax.nn.softmax(log_weights_trace[-1])
        idx = jnp.searchsorted(jnp.cumsum(w), jax.random.uniform(key))
        state_traj = reconstruct_trajectory(state_trace, ancestor_trace, idx)

        return state_traj


class PGAS(eqx.Module):
    N_iterations: int
    N_steps: int
    cSMC: condSequentialMonteCarlo
    GP_prior: tuple[jax.Array]

    def __init__(
        self,
        N_samples: int,
        N_iterations: int,
        observations: jax.Array,
        inputs: jax.Array,
        init_state_mean: jax.Array,
        init_state_cov: jax.Array,
        likelihood_fcn: Callable[[jax.Array, jax.Array], jax.Array],
        GP_prior: tuple[jax.Array],
        basis_fcn: Callable[[jax.Array, jax.Array], jax.Array],
    ):
        self.N_iterations = N_iterations
        self.N_steps = observations.shape[0]
        self.GP_prior = GP_prior
        self.cSMC = condSequentialMonteCarlo(
            N_samples=N_samples,
            observations=observations,
            inputs=inputs,
            init_state_mean=init_state_mean,
            init_state_cov=init_state_cov,
            likelihood_fcn=likelihood_fcn,
            basis_fcn=basis_fcn,
        )

    def _init_algorithm(
        self,
        init_ref_state: jax.Array,
    ) -> tuple[jax.Array]:
        state_trace = np.zeros(
            (
                self.N_iterations,
                self.N_steps,
                self.cSMC.init_state_mean.shape[0],
            )
        )
        state_trace[0] = np.atleast_2d(init_ref_state.T).T

        # trace for coefficient matrix
        coeff_mat = np.zeros(
            (
                self.GP_prior[0].shape[1],
                self.GP_prior[0].shape[0],
            )
        )

        # trace for error cov matrix
        error_cov_mat = np.zeros((self.N_iterations, *self.GP_prior[2].shape))

        return state_trace, coeff_mat, error_cov_mat

    def sample_params(
        self,
        key: jax.Array,
        state_trajectory: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # calculate sufficient statistics
        basis = jax.vmap(self.cSMC.basis_fcn)(
            state_trajectory[:-1], self.cSMC.inputs[:-1]
        )
        T = jax.vmap(BI.prior_mniw_calcStatistics)(state_trajectory[1:], basis)
        suff_stats = (
            self.GP_prior[0] + jnp.sum(T[0], axis=0),
            self.GP_prior[1] + jnp.sum(T[1], axis=0),
            self.GP_prior[2] + jnp.sum(T[2], axis=0),
            self.GP_prior[3] + jnp.sum(T[3], axis=0),
        )

        # convert to standard parameters
        mean, col_cov, row_scale, df = BI.prior_mniw_2naturalPara_inv(
            *suff_stats
        )

        ## sample parameters
        key, key_A, key_S = jax.random.split(key, 3)

        # Sample S ~ InvWishart(df, row_scale) via
        # Axen 2023 (https://arxiv.org/pdf/2310.15884v1)
        p = row_scale.shape[0]
        # Compute L such that L L^T = row_scale^{-1} without inverting explicitly
        chol_row = jnp.linalg.cholesky(row_scale)
        identity_matrix = jnp.eye(p, dtype=row_scale.dtype)
        L = jsp.linalg.solve_triangular(chol_row, identity_matrix, lower=True)

        # Decomposition: T lower triangular
        # Diagonal ~ sqrt(Chi2_{df - i}), i = 0..p-1; below-diagonal ~ N(0,1)
        nu = df - jnp.arange(p, dtype=df.dtype)
        # Ensure positivity in case of small numerical issues
        # nu = jnp.clip(nu, a_min=1e-6)
        key_S, key_norm = jax.random.split(key_S)
        diag_samples = jnp.sqrt(jax.random.chisquare(key_S, nu))
        normals = jax.random.normal(key_norm, (p, p))
        T = jnp.tril(normals, k=-1) + jnp.diag(diag_samples)

        # Cholesky of W ~ Wishart(df, row_scale^{-1}) is C = L @ T (lower triangular)
        C = L @ T
        # Then S = W^{-1} has Cholesky factor S_chol = C^{-T}
        S_chol = jsp.linalg.solve_triangular(C.T, identity_matrix, lower=False)
        S = S_chol @ S_chol.T

        # sample from multivariate normal
        N = jax.random.normal(key_A, mean.shape)
        V_chol = jnp.linalg.cholesky(col_cov)
        # S_chol already computed above
        A = mean + jnp.einsum("ij,jk,kl->il", S_chol, N, V_chol)

        return A, S

    def __call__(
        self,
        key: jax.Array,
        init_ref_state: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array]]:
        # initialize
        state_trace, coeff_mat, error_cov_mat = self._init_algorithm(
            init_ref_state
        )

        # sample initial parameters
        key, key_para = jax.random.split(key, 2)
        jit_sample_params = eqx.filter_jit(self.sample_params)
        coeff_mat, error_cov_mat = jit_sample_params(key_para, state_trace[0])

        # run PGAS steps
        for k in tqdm(
            range(1, self.N_iterations), desc="Running PGAS iterations"
        ):
            # run cSMC (draw a new trajectory)
            key, key_step = jax.random.split(key)
            new_state = self.cSMC(
                key_step,
                state_trace[k - 1],
                coeff_mat,
                error_cov_mat,
            )

            # save draw in trace
            state_trace[k] = np.atleast_2d(new_state.T).T

            # sample new parameters
            key, key_para = jax.random.split(key, 2)
            coeff_mat, error_cov_mat = jit_sample_params(key_para, new_state)

        state_trace = np.swapaxes(state_trace, 0, 1)

        # calculate log likelihood
        log_likelihood = jax.vmap(
            jax.vmap(
                self.cSMC.likelihood_fcn,
                in_axes=(
                    None,
                    0,
                    None,
                ),
            )
        )(self.cSMC.observations, state_trace, self.cSMC.inputs)

        return (
            state_trace,
            log_likelihood,
        )
