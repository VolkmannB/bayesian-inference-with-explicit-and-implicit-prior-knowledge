import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
from typing import Callable


class StateSpaceModel(eqx.Module):
    process_noise: jax.Array
    output_noise: jax.Array
    transition_model: Callable[
        [jax.Array, jax.Array, tuple[jax.Array, ...], dict], jax.Array
    ]
    output_model: Callable[
        [jax.Array, jax.Array, tuple[jax.Array, ...], dict], jax.Array
    ]
    is_deterministic: bool

    def __init__(
        self,
        process_noise: jax.Array,
        output_noise: jax.Array,
        transition_model: Callable,
        output_model: Callable,
    ):
        self.process_noise = jnp.array(process_noise)
        self.output_noise = jnp.array(output_noise)
        self.transition_model = transition_model
        self.output_model = output_model
        self.is_deterministic = bool(jnp.all(process_noise == 0))

    def transition_mdl(
        self,
        state: jax.Array,
        input: jax.Array,
        *int_variables: tuple[jax.Array],
    ) -> jax.Array:
        return self.transition_model(
            state,
            input,
            *int_variables,
        )

    def output_mdl(
        self,
        state: jax.Array,
        input: jax.Array,
        *int_variables: tuple[jax.Array],
    ) -> jax.Array:
        return self.output_model(
            state,
            input,
            *int_variables,
        )

    def draw_state(
        self,
        key: jax.Array,
        state: jax.Array,
        input: jax.Array,
        *int_variables: tuple[jax.Array],
    ) -> jax.Array:
        # evaluate model
        new_state = self.transition_mdl(state, input, *int_variables)

        # sample Gaussian noise
        std_normal = jax.random.normal(key, shape=state.shape)
        Cov_chol = jnp.linalg.cholesky(self.process_noise)

        if self.is_deterministic:
            return new_state
        else:
            return new_state + Cov_chol @ std_normal

    def log_likelihood(
        self,
        observation: jax.Array,
        state: jax.Array,
        input: jax.Array,
        *int_variables: tuple[jax.Array],
    ) -> float:
        output = self.output_mdl(state, input, *int_variables)
        return jsp.stats.multivariate_normal.logpdf(
            observation,
            mean=jnp.atleast_1d(output),
            cov=jnp.atleast_2d(self.output_noise),
        )
