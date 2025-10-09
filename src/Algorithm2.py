import jax
import numpy as np
from typing import Callable
from tqdm import tqdm

from src.StateSpaceModel import StateSpaceModel
import src.BayesianInferrence as BI
from src.Algorithm3 import Algorithm3


class Algorithm2:
    def __init__(
        self,
        N_samples: int,
        N_iterations: int,
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
        self.N_iterations = N_iterations
        self.N_steps = observations.shape[0]
        self.cSMC = Algorithm3(
            N_samples=N_samples,
            observations=observations,
            inputs=inputs,
            SSM=SSM,
            init_state_mean=init_state_mean,
            init_state_cov=init_state_cov,
            init_int_var_mean=init_int_var_mean,
            init_int_var_cov=init_int_var_cov,
            GP_prior=GP_prior,
            basis_fcn=basis_fcn,
        )

    def _init_algorithm(
        self,
        init_ref_state: jax.Array,
        init_ref_int_var: tuple[jax.Array],
    ) -> tuple[jax.Array]:
        state_trace = np.zeros(
            (
                self.N_iterations,
                self.N_steps,
                self.cSMC.init_state_mean.shape[0],
            )
        )
        state_trace[0] = init_ref_state

        int_var_trace = [
            np.zeros(
                (
                    self.N_iterations,
                    self.N_steps,
                    self.cSMC.init_int_var_mean[i].shape[0],
                )
            )
            for i in range(len(self.cSMC.init_int_var_mean))
        ]
        for i in range(len(self.cSMC.init_int_var_mean)):
            int_var_trace[i][0] = np.atleast_2d(init_ref_int_var[i].T).T

        suff_stats_trace = [
            [
                np.zeros((self.N_iterations, *self.cSMC.GP_prior[i][0].shape)),
                np.zeros(
                    (
                        self.N_iterations,
                        *self.cSMC.GP_prior[i][1].shape,
                    )
                ),
                np.zeros((self.N_iterations, *self.cSMC.GP_prior[i][2].shape)),
                np.zeros((self.N_iterations, *self.cSMC.GP_prior[i][3].shape)),
            ]
            for i in range(self.cSMC.dim_basis.shape[0])
        ]

        # set initial reference sufficient statistics
        ref_basis = [
            jax.vmap(self.cSMC.basis_fcn[i])(init_ref_state, self.cSMC.inputs)
            for i in range(len(self.cSMC.basis_fcn))
        ]
        ref_suff_stats = [
            jax.vmap(BI.prior_mniw_calcStatistics)(
                init_ref_int_var[i], ref_basis[i]
            )
            for i in range(len(init_ref_int_var))
        ]
        ref_suff_stats = [
            [np.sum(ref_suff_stats[i][j], axis=0) for j in range(4)]
            for i in range(len(init_ref_int_var))
        ]
        for i in range(len(self.cSMC.dim_basis)):
            suff_stats_trace[i][0][0] = ref_suff_stats[i][0]
            suff_stats_trace[i][1][0] = ref_suff_stats[i][1]
            suff_stats_trace[i][2][0] = ref_suff_stats[i][2]
            suff_stats_trace[i][3][0] = ref_suff_stats[i][3]

        return state_trace, int_var_trace, suff_stats_trace

    def __call__(
        self,
        key: jax.Array,
        init_ref_state: jax.Array,
        init_ref_int_var: tuple[jax.Array],
    ) -> tuple[jax.Array]:
        # initialize
        state_trace, int_var_trace, suff_stats_trace = self._init_algorithm(
            init_ref_state, init_ref_int_var
        )

        for k in tqdm(
            range(1, self.N_iterations), desc="Running PGAS iterations"
        ):
            # run Algorithm 2 (draw a new trajectory)
            key, key_step = jax.random.split(key)
            (
                new_state,
                new_int_var,
            ) = self.cSMC(
                key_step,
                state_trace[k - 1],
                [int_var_trace[i][k - 1] for i in range(len(int_var_trace))],
                [
                    [suff_stats_trace[i][j][k - 1] for j in range(4)]
                    for i in range(len(suff_stats_trace))
                ],
            )

            # save draw in trace
            state_trace[k] = np.atleast_2d(new_state.T).T
            for i in range(len(int_var_trace)):
                int_var_trace[i][k] = np.atleast_2d(new_int_var[i].T).T
                new_basis = jax.vmap(self.cSMC.basis_fcn[i])(
                    new_state, self.cSMC.inputs
                )
                ref_suff_stats = jax.vmap(BI.prior_mniw_calcStatistics)(
                    new_int_var[i], new_basis
                )
                ref_suff_stats = [
                    np.sum(ref_suff_stats[j], axis=0) for j in range(4)
                ]

                suff_stats_trace[i][0][k] = ref_suff_stats[0]
                suff_stats_trace[i][1][k] = ref_suff_stats[1]
                suff_stats_trace[i][2][k] = ref_suff_stats[2]
                suff_stats_trace[i][3][k] = ref_suff_stats[3]
        state_trace = np.swapaxes(state_trace, 0, 1)
        int_var_trace = [
            np.swapaxes(int_var_trace[i], 0, 1)
            for i in range(len(int_var_trace))
        ]

        # calculate observations
        obs_trace = jax.vmap(
            jax.vmap(
                self.cSMC.SSM.output_mdl,
                in_axes=(0, None, *[0 for _ in range(len(int_var_trace))]),
            )
        )(state_trace, self.cSMC.inputs, *int_var_trace)

        # calculate log likelihood
        log_likelihood = jax.vmap(
            jax.vmap(
                self.cSMC.SSM.log_likelihood,
                in_axes=(
                    None,
                    0,
                    None,
                    *[0 for _ in range(len(int_var_trace))],
                ),
            )
        )(self.cSMC.observations, state_trace, self.cSMC.inputs, *int_var_trace)

        return (
            state_trace,
            int_var_trace,
            np.ones((self.N_steps, self.N_iterations)) / self.N_iterations,
            suff_stats_trace,
            obs_trace,
            log_likelihood,
        )
