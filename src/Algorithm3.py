import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
from typing import Callable
from tqdm import tqdm

from src.StateSpaceModel import StateSpaceModel
import src.BayesianInferrence as BI
import src.Filtering as Filtering
from src.Filtering import reconstruct_trajectory
from src.Algorithm1 import Algorithm1


class Algorithm3(Algorithm1):
    def __init__(
        self,
        N_samples: int,
        observations: jax.Array,
        inputs: jax.Array,
        SSM: StateSpaceModel,
        init_state_mean: jax.Array,
        init_state_cov: jax.Array,
        init_int_var_mean: tuple[jax.Array],
        init_int_var_cov: tuple[jax.Array],
        GP_prior: tuple[tuple[jax.Array]],
        basis_fcn: tuple[Callable[[jax.Array, jax.Array], jax.Array]],
    ):
        super().__init__(
            N_samples,
            observations,
            inputs,
            SSM,
            1.0,
            init_state_mean,
            init_state_cov,
            init_int_var_mean,
            init_int_var_cov,
            GP_prior,
            basis_fcn,
        )

    def step(
        self,
        key: jax.Array,
        time: int,
        log_weights: jax.Array,
        state: jax.Array,
        int_var: tuple[jax.Array],
        suff_stats: tuple[tuple[jax.Array]],
        ref_state: jax.Array,
        ref_int_var: tuple[jax.Array],
        ref_suff_stats: tuple[tuple[jax.Array]],
    ) -> tuple[
        jax.Array,
        jax.Array,
        tuple[jax.Array],
        tuple[tuple[jax.Array]],
        jax.Array,
        tuple[tuple[jax.Array]],
    ]:
        N_int_var = self.dim_basis.shape[0]

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

        # draw ancestor for reference trajectory
        g_T = jnp.zeros(self.N_samples)
        g_t = jnp.zeros(self.N_samples)
        for i in range(N_int_var):
            g_T += jax.vmap(BI.prior_mniw_log_base_measure)(
                self.GP_prior[i][0] + ref_suff_stats[i][0] + suff_stats[i][0],
                self.GP_prior[i][1] + ref_suff_stats[i][1] + suff_stats[i][1],
                self.GP_prior[i][2] + ref_suff_stats[i][2] + suff_stats[i][2],
                self.GP_prior[i][3] + ref_suff_stats[i][3] + suff_stats[i][3],
            )
            g_t += jax.vmap(BI.prior_mniw_log_base_measure)(
                self.GP_prior[i][0] + suff_stats[i][0],
                self.GP_prior[i][1] + suff_stats[i][1],
                self.GP_prior[i][2] + suff_stats[i][2],
                self.GP_prior[i][3] + suff_stats[i][3],
            )
        h_x = jax.vmap(
            jsp.stats.multivariate_normal.logpdf,
            in_axes=(None, 0, None),
        )(
            ref_state,
            aux_state,
            self.SSM.process_noise,
        )
        log_weights_ancestor = log_weights_aux + g_t - g_T + h_x
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
            key_state, time, state, int_var, a_indices
        )
        new_state = new_state.at[-1].set(ref_state)

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
        new_int_var = tuple(
            new_int_var[i].at[-1].set(jnp.squeeze(ref_int_var[i]))
            for i in range(N_int_var)
        )

        # update sufficient statistics
        T = tuple(
            jax.vmap(BI.prior_mniw_calcStatistics)(new_int_var[i], new_basis[i])
            for i in range(N_int_var)
        )
        new_suff_stats = tuple(
            tuple(suff_stats_args[i][j] + T[i][j] for j in range(4))
            for i in range(N_int_var)
        )

        # update reference sufficient statistics
        ref_basis = tuple(
            self.basis_fcn[i](ref_state, self.inputs[time])
            for i in range(N_int_var)
        )
        ref_T = tuple(
            BI.prior_mniw_calcStatistics(ref_int_var[i], ref_basis[i])
            for i in range(N_int_var)
        )
        ref_suff_stats = tuple(
            tuple(ref_suff_stats[i][j] - ref_T[i][j] for j in range(4))
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
            ref_suff_stats,
        )

    def __call__(
        self,
        key: jax.Array,
        ref_state: jax.Array,
        ref_int_var: tuple[jax.Array],
        ref_suff_stats: tuple[tuple[jax.Array]],
    ) -> tuple[jax.Array]:
        N_int_var = self.dim_basis.shape[0]
        # run initialization
        key, key_init = jax.random.split(key)
        (
            state_trace,
            int_var_trace,
            _,
            log_weights_trace,
            ancestor_trace,
            suff_stats,
        ) = self._init_algorithm(
            key_init,
        )

        # set initial reference trajectory
        state_trace[0, -1] = ref_state[0]
        new_suff_stats = list(suff_stats)
        for i in range(N_int_var):
            int_var_trace[i][0, -1] = ref_int_var[i][0]
            init_basis = self.basis_fcn[i](ref_state[0], self.inputs[0])
            init_T = BI.prior_mniw_calcStatistics(ref_int_var[i][0], init_basis)

            new_suff_stats_i = list(new_suff_stats[i])
            for j in range(4):
                new_suff_stats_i[j] = new_suff_stats[i][j].at[-1].set(init_T[j])
            new_suff_stats[i] = tuple(new_suff_stats_i)
        suff_stats = tuple(new_suff_stats)

        # set initial reference sufficient statistics
        ref_basis = tuple(
            self.basis_fcn[i](ref_state[0], self.inputs[0])
            for i in range(N_int_var)
        )
        ref_T = tuple(
            BI.prior_mniw_calcStatistics(ref_int_var[i][0], ref_basis[i])
            for i in range(N_int_var)
        )
        ref_suff_stats = tuple(
            tuple(ref_suff_stats[i][j] - ref_T[i][j] for j in range(4))
            for i in range(N_int_var)
        )

        # run loop
        jit_step = eqx.filter_jit(self.step)
        N_steps = self.observations.shape[0]
        for time in tqdm(
            range(1, N_steps), desc="Processing time steps", leave=False
        ):
            # run a step of the loop in Algorithm 3
            key, key_step = jax.random.split(key)
            # Pre-compute arguments to avoid dynamic list creation
            int_var_args = tuple(
                int_var_trace[i][time - 1] for i in range(N_int_var)
            )
            ref_int_var_args = tuple(
                ref_int_var[i][time] for i in range(N_int_var)
            )

            (
                new_log_weights,
                new_state,
                new_int_var,
                new_suff_stats,
                a_indices,
                ref_suff_stats,
            ) = jit_step(
                key_step,
                jnp.array(time),
                log_weights_trace[time - 1],
                state_trace[time - 1],
                int_var_args,
                suff_stats,
                ref_state[time],
                ref_int_var_args,
                ref_suff_stats,
            )

            # save results in trace
            state_trace[time] = new_state
            log_weights_trace[time] = new_log_weights
            ancestor_trace[time - 1] = a_indices
            for i in range(N_int_var):
                int_var_trace[i][time] = jnp.atleast_2d(new_int_var[i].T).T
            suff_stats = new_suff_stats

        # sample a trajectory
        w = jax.nn.softmax(log_weights_trace[-1])
        idx = jnp.searchsorted(jnp.cumsum(w), jax.random.uniform(key))
        state_traj = reconstruct_trajectory(state_trace, ancestor_trace, idx)
        int_var_traj = tuple(
            reconstruct_trajectory(int_var_trace[i], ancestor_trace, idx)
            for i in range(N_int_var)
        )

        return (
            state_traj,
            int_var_traj,
        )
