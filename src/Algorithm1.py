import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Callable
from tqdm import tqdm

from src.StateSpaceModel import StateSpaceModel
import src.BayesianInferrence as BI
import src.Filtering as Filtering


class Algorithm1(eqx.Module):
    N_samples: int
    observations: jax.Array
    inputs: jax.Array
    SSM: StateSpaceModel
    forgetting_factor: float
    init_state_mean: jax.Array
    init_state_cov: jax.Array
    init_int_var_mean: tuple[jax.Array]
    init_int_var_cov: tuple[jax.Array]
    GP_prior: tuple[tuple[jax.Array]]
    basis_fcn: tuple[Callable[[jax.Array, jax.Array], jax.Array]]
    dim_basis: jax.Array

    def __init__(
        self,
        N_samples: int,
        observations: jax.Array,
        inputs: jax.Array,
        SSM: StateSpaceModel,
        forgetting_factor: float,
        init_state_mean: jax.Array,
        init_state_cov: jax.Array,
        init_int_var_mean: tuple[jax.Array],
        init_int_var_cov: tuple[jax.Array],
        GP_prior: tuple[tuple[jax.Array]],
        basis_fcn: tuple[Callable[[jax.Array, jax.Array], jax.Array]],
    ):
        self.N_samples = N_samples
        self.observations = jnp.array(observations)
        self.inputs = jnp.array(inputs)
        self.SSM = SSM
        self.forgetting_factor = forgetting_factor
        self.init_state_cov = jnp.array(init_state_cov)
        self.init_state_mean = jnp.array(init_state_mean)
        self.init_int_var_mean = [
            jnp.array(init_int_var_mean[i])
            for i in range(len(init_int_var_mean))
        ]
        self.init_int_var_cov = [
            jnp.array(init_int_var_cov[i]) for i in range(len(init_int_var_cov))
        ]
        self.basis_fcn = basis_fcn
        self.dim_basis = jnp.array(
            [
                len(self.basis_fcn[i](self.init_state_mean, self.inputs[0]))
                for i in range(len(self.basis_fcn))
            ],
            dtype=jnp.int32,
        )
        self.GP_prior = [
            [jnp.array(GP_prior[i][j]) for j in jnp.arange(4)]
            for i in range(len(GP_prior))
        ]

    def _init_trace_vars(self) -> tuple[jax.Array]:
        N_steps = self.observations.shape[0]
        state_trace = np.zeros(
            (N_steps, self.N_samples, self.init_state_mean.shape[0])
        )
        int_var_trace = [
            np.zeros(
                (N_steps, self.N_samples, self.init_int_var_mean[i].shape[0])
            )
            for i in jnp.arange(self.dim_basis.shape[0])
        ]
        suff_stats_trace = [
            [
                np.zeros((N_steps, *self.GP_prior[i][0].shape)),
                np.zeros((N_steps, *self.GP_prior[i][1].shape)),
                np.zeros((N_steps, *self.GP_prior[i][2].shape)),
                np.zeros((N_steps, *self.GP_prior[i][3].shape)),
            ]
            for i in range(self.dim_basis.shape[0])
        ]
        log_weights_trace = np.zeros((N_steps, self.N_samples))
        ancestor_trace = np.zeros(
            (N_steps - 1, self.N_samples), dtype=jnp.int32
        )
        return (
            state_trace,
            int_var_trace,
            suff_stats_trace,
            log_weights_trace,
            ancestor_trace,
        )

    def _init_algorithm(
        self,
        key: jax.Array,
    ) -> tuple[
        jax.Array,  # state_trace
        list[jax.Array],  # int_var_trace
        list[list[jax.Array]],  # suff_stats_trace
        jax.Array,  # log_weights_trace
        jax.Array,  # ancestor_trace
        tuple[tuple[jax.Array]],  # suff_stats
    ]:
        """
        Initialize the algorithm by generating initial particles, weights, and sufficient statistics.

        Args:
            key: JAX random key for stochastic operations

        Returns:
            Tuple containing:
            - state_trace: Trace of particle states
            - int_var_trace: Trace of interface variables
            - suff_stats_trace: Trace of sufficient statistics
            - log_weights_trace: Trace of log weights
            - ancestor_trace: Trace of ancestors
            - suff_stats: Current sufficient statistics
        """
        # initialize variables
        (
            state_trace,
            int_var_trace,
            suff_stats_trace,
            log_weights_trace,
            ancestor_trace,
        ) = self._init_trace_vars()

        # generate initial particles
        key, key_state = jax.random.split(key)
        state_trace[0] = jax.random.multivariate_normal(
            key_state,
            self.init_state_mean,
            self.init_state_cov,
            (self.N_samples,),
        )
        for i in jnp.arange(self.dim_basis.shape[0]):
            key, key_aux = jax.random.split(key)
            int_var_trace[i][0] = jax.random.multivariate_normal(
                key_aux,
                self.init_int_var_mean[i],
                self.init_int_var_cov[i],
                (self.N_samples,),
            )

        # calculate sufficient statistics
        suff_stats = []
        for i in jnp.arange(self.dim_basis.shape[0]):
            basis = jax.vmap(self.basis_fcn[i], in_axes=(0, None))(
                state_trace[0], self.inputs[0]
            )
            T = jax.vmap(BI.prior_mniw_calcStatistics)(
                int_var_trace[i][0], basis
            )
            suff_stats.append(T)

            w = jax.nn.softmax(log_weights_trace[0])
            suff_stats_trace[i][0][0] = jnp.einsum("j...,j->...", T[0], w)
            suff_stats_trace[i][1][0] = jnp.einsum("j...,j->...", T[1], w)
            suff_stats_trace[i][2][0] = jnp.einsum("j...,j->...", T[2], w)
            suff_stats_trace[i][3][0] = jnp.einsum("j...,j->...", T[3], w)
        suff_stats = tuple(suff_stats)

        return (
            state_trace,
            int_var_trace,
            suff_stats_trace,
            log_weights_trace,
            ancestor_trace,
            suff_stats,
        )

    def _generate_auxiliary_states(
        self,
        state: jax.Array,
        time: int,
        int_var: tuple[jax.Array],
        suff_stats: tuple[tuple[jax.Array]],
    ) -> tuple[jax.Array, tuple[jax.Array]]:
        """
        Generate auxiliary states for the auxiliary particle filter.

        Args:
            state: Current state particles
            time: Current time step
            int_var: Interface variables
            suff_stats: Sufficient statistics

        Returns:
            Tuple containing:
            - aux_state: Auxiliary states
            - aux_int_var: Auxiliary interface variables
        """
        # Generate auxiliary states
        # Create proper in_axes - map over state and all interface variables
        in_axes = (0, None) + tuple(
            0 for _ in jnp.arange(self.dim_basis.shape[0])
        )

        aux_state = jax.vmap(self.SSM.transition_mdl, in_axes=in_axes)(
            state, self.inputs[time - 1], *int_var
        )

        # Calculate auxiliary GP means
        aux_GP_mean = tuple(
            jax.vmap(BI.prior_mniw_mean)(
                suff_stats[i][0] + self.GP_prior[i][0],
                suff_stats[i][1] + self.GP_prior[i][1],
            )
            for i in range(self.dim_basis.shape[0])
        )

        # Evaluate basis functions at auxiliary states
        aux_basis = tuple(
            jax.vmap(self.basis_fcn[i], in_axes=(0, None))(
                aux_state, self.inputs[time]
            )
            for i in range(self.dim_basis.shape[0])
        )

        # Calculate auxiliary interface variables
        aux_int_var = tuple(
            jnp.einsum("ikj,ij->ik", aux_GP_mean[i], aux_basis[i])
            for i in range(self.dim_basis.shape[0])
        )

        return aux_state, aux_int_var

    def _draw_int_vars(
        self,
        key: jax.Array,
        time: int,
        state: jax.Array,
        suff_stats: tuple[tuple[jax.Array]],
    ) -> tuple[tuple[jax.Array], tuple[jax.Array]]:
        N_int_var = self.dim_basis.shape[0]
        # evaluate basis funtcions
        basis = tuple(
            jax.vmap(self.basis_fcn[i], in_axes=(0, None))(
                state, self.inputs[time]
            )
            for i in range(N_int_var)
        )

        # transform sufficient statistics into standard parameters
        std_para = tuple(
            jax.vmap(BI.prior_mniw_2naturalPara_inv)(
                *tuple(suff_stats[i][j] + self.GP_prior[i][j] for j in range(4))
            )
            for i in range(N_int_var)
        )

        # calculate predictive distribution
        pred_dist = tuple(
            jax.vmap(BI.prior_mniw_Predictive)(*std_para[i], basis[i])
            for i in range(N_int_var)
        )

        # draw from predictive distribution
        keys = jax.random.split(key, N_int_var)
        int_var = tuple(
            jax.vmap(BI.prior_mniw_drawPred)(
                jax.random.split(keys[i], self.N_samples), *pred_dist[i]
            )
            for i in range(N_int_var)
        )

        return int_var, basis

    def _draw_states(
        self,
        key: jax.Array,
        time: int,
        state: jax.Array,
        int_var: tuple[jax.Array],
        a_indices: jax.Array,
    ) -> tuple[jax.Array]:
        N_int_var = self.dim_basis.shape[0]
        # Pre-compute int_var arguments to avoid dynamic unpacking
        int_var_args = tuple(int_var[i][a_indices] for i in range(N_int_var))
        new_state = jax.vmap(
            self.SSM.draw_state,
            in_axes=(0, 0, None, *[0 for _ in range(N_int_var)]),
        )(
            jax.random.split(key, self.N_samples),
            state[a_indices],
            self.inputs[time - 1],
            *int_var_args,
        )
        return new_state

    def step(
        self,
        key: jax.Array,
        time: int,
        log_weights: jax.Array,
        state: jax.Array,
        int_var: tuple[jax.Array],
        suff_stats: tuple[tuple[jax.Array]],
    ) -> tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array],
        tuple[tuple[jax.Array]],
        jax.Array,
    ]:
        N_int_var = self.dim_basis.shape[0]
        # statistics time update
        suff_stats = tuple(
            tuple(suff_stats[i][j] * self.forgetting_factor for j in range(4))
            for i in range(N_int_var)
        )

        # generate auxiliary states
        aux_state, aux_int_var = self._generate_auxiliary_states(
            state, time, int_var, suff_stats
        )

        # compute first stage weights
        log_likelihood_aux = jax.vmap(
            self.SSM.log_likelihood,
            in_axes=(
                None,
                0,
                None,
                *[0 for _ in jnp.arange(N_int_var)],
            ),
        )(
            self.observations[time],
            aux_state,
            self.inputs[time],
            *aux_int_var,
        )
        log_weights_aux = log_likelihood_aux + log_weights
        aux_weights = jax.nn.softmax(log_weights_aux)

        # draw ancestors
        key, key_ancestor = jax.random.split(key)
        a_indices = Filtering.systematic_SISR(key_ancestor, aux_weights)

        # draw new state
        key, key_state = jax.random.split(key)
        new_state = self._draw_states(
            key_state, time, state, int_var, a_indices
        )

        # draw new interface variables
        key, key_int = jax.random.split(key)
        # Pre-compute suff_stats arguments to avoid dynamic list creation
        suff_stats_args = tuple(
            tuple(suff_stats[i][j][a_indices] for j in range(4))
            for i in range(N_int_var)
        )
        new_int_var, new_basis = self._draw_int_vars(
            key_int,
            time,
            new_state,
            suff_stats_args,
        )

        # statistics measurement update
        T = tuple(
            jax.vmap(BI.prior_mniw_calcStatistics)(new_int_var[i], new_basis[i])
            for i in range(N_int_var)
        )
        new_suff_stats = tuple(
            tuple(suff_stats_args[i][j] + T[i][j] for j in range(4))
            for i in range(N_int_var)
        )

        # compute new weights
        new_log_weights = (
            jax.vmap(
                self.SSM.log_likelihood,
                in_axes=(None, 0, None, *[0 for _ in range(N_int_var)]),
            )(
                self.observations[time],
                new_state,
                self.inputs[time],
                *new_int_var,
            )
            - log_likelihood_aux[a_indices]
        )

        return (
            new_log_weights,
            new_state,
            new_int_var,
            new_suff_stats,
            a_indices,
        )

    def __call__(
        self,
        key: jax.Array,
    ) -> tuple[jax.Array]:
        # run initialization
        key, key_init = jax.random.split(key)
        (
            state_trace,
            int_var_trace,
            suff_stats_trace,
            log_weights_trace,
            ancestor_trace,
            suff_stats,
        ) = self._init_algorithm(
            key_init,
        )

        # run loop
        jit_step = eqx.filter_jit(self.step)
        N_steps = self.observations.shape[0]
        N_int_var = self.dim_basis.shape[0]
        for time in tqdm(range(1, N_steps), desc="Processing time steps"):
            # run a step of the loop in Algorithm 1
            key, key_step = jax.random.split(key)
            (
                new_log_weights,
                new_state,
                new_int_var,
                new_suff_stats,
                a_indices,
            ) = jit_step(
                key_step,
                jnp.array(time),
                log_weights_trace[time - 1],
                state_trace[time - 1],
                [int_var_trace[i][time - 1] for i in range(N_int_var)],
                suff_stats,
            )

            # save results in trace
            state_trace[time] = new_state
            log_weights_trace[time] = new_log_weights
            ancestor_trace[time - 1] = a_indices
            for i in range(N_int_var):
                int_var_trace[i][time] = jnp.atleast_2d(new_int_var[i].T).T

                w = jax.nn.softmax(new_log_weights)
                suff_stats_trace[i][0][time] = np.einsum(
                    "n...,n->...", new_suff_stats[i][0], w
                )
                suff_stats_trace[i][1][time] = np.einsum(
                    "n...,n->...", new_suff_stats[i][1], w
                )
                suff_stats_trace[i][2][time] = np.einsum(
                    "n...,n->...", new_suff_stats[i][2], w
                )
                suff_stats_trace[i][3][time] = np.einsum(
                    "n...,n->...", new_suff_stats[i][3], w
                )
            suff_stats = new_suff_stats

        weights_trace = jax.nn.softmax(log_weights_trace, axis=1)

        # calculate observations
        obs_trace = jax.vmap(
            jax.vmap(
                self.SSM.output_mdl,
                in_axes=(0, None, *[0 for _ in range(N_int_var)]),
            )
        )(state_trace, self.inputs, *int_var_trace)

        # calculate log likelihood
        log_likelihood = jax.vmap(
            jax.vmap(
                self.SSM.log_likelihood,
                in_axes=(None, 0, None, *[0 for _ in range(N_int_var)]),
            )
        )(
            self.observations,
            state_trace,
            self.inputs,
            *int_var_trace,
        )

        return (
            state_trace,
            int_var_trace,
            suff_stats_trace,
            weights_trace,
            ancestor_trace,
            suff_stats,
            obs_trace,
            log_likelihood,
        )
