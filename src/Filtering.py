import numpy as np
import jax
import jax.numpy as jnp


def systematic_SISR(key: jax.Array, w: jax.Array):
    """
    Systematic resampling for Sequential Importance Sampling with Resampling
    (SISR).

    Args:
        key (jax.random.PRNGKey): Random number generator key
        w (ArrayLike): Unnormalized weights

    Returns:
        ArrayLike: Resampled indices
    """
    # number of samples
    u = jax.random.uniform(key)
    N = len(w)

    # Ensure valid input
    w = jnp.clip(w, 0.0, jnp.inf)  # Ensure non-negative weights
    w_sum = jnp.sum(w)
    w = jnp.where(w_sum > 0, w / w_sum, jnp.ones_like(w) / N)

    # select deterministic samples
    U = (u + jnp.arange(N)) / N
    W = jnp.cumsum(w)
    W = jnp.clip(
        W, 0.0, 1.0
    )  # Ensure cumsum doesn't exceed 1 due to numerical errors

    indices = jnp.searchsorted(W, U)
    indices = jnp.clip(indices, 0, N - 1)  # Ensure valid indices

    return indices


def reconstruct_trajectory(Particles, ancestry, idx):
    Particles = np.atleast_3d(Particles)

    n_steps = Particles.shape[0]
    n_dim = Particles.shape[-1]
    traj = np.zeros((n_steps, n_dim))

    ancestor_idx = np.zeros((n_steps,))
    ancestor_idx[-1] = idx

    traj[-1] = Particles[-1, idx]
    for i in range(n_steps - 2, -1, -1):  # run backward in time
        ancestor_idx[i] = ancestry[i, int(ancestor_idx[i + 1])]
        traj[i] = Particles[i, int(ancestor_idx[i])]

    return np.squeeze(traj)
