import numpy as np
import functools
import jax
import jax.numpy as jnp
import heapq


def generate_Hilbert_BasisFunction(
    num_fcn, domain_boundary, lengthscale, scale, idx_start=1, idx_step=1
):
    # dimensionality of the input
    domain_boundary = np.atleast_2d(domain_boundary)
    num_dims = domain_boundary.shape[0]

    # center domain L
    domain_center = (domain_boundary[:, 0] + domain_boundary[:, 1]) / 2

    # set start index to 1 if value is negative
    if idx_start < 1:
        idx_start = 1

    # set indices per dimension
    domain_size = domain_boundary[:, 1] - domain_boundary[:, 0]
    idx_end = num_fcn * idx_step + 1 + idx_start
    j = np.arange(idx_start, idx_end, idx_step)

    # efficiently find only the num_fcn combinations with the smallest eigenvalues
    # without materializing the full Cartesian product
    weights = (np.pi / domain_size) ** 2  # per-dimension weight
    j_sq = j**2

    # best-first search over the index lattice using a min-heap
    start_idx = tuple([0] * num_dims)
    start_cost = float(np.sum(weights * j_sq[0]))
    heap = [(start_cost, start_idx)]
    visited = {start_idx}
    selected = []

    while len(selected) < num_fcn and heap:
        cost, idx_tuple = heapq.heappop(heap)
        s_vec = j[np.array(idx_tuple, dtype=int)]
        selected.append(s_vec)

        # explore neighbors by incrementing a single dimension
        for d in range(num_dims):
            if idx_tuple[d] + 1 < len(j):
                neighbor = list(idx_tuple)
                old_i = neighbor[d]
                neighbor[d] = old_i + 1
                neighbor_t = tuple(neighbor)
                if neighbor_t not in visited:
                    # incremental cost update (monotone in each coordinate)
                    new_cost = cost + float(
                        weights[d] * (j_sq[neighbor[d]] - j_sq[old_i])
                    )
                    heapq.heappush(heap, (new_cost, neighbor_t))
                    visited.add(neighbor_t)

    S_selected = np.array(selected, dtype=float)
    eig_val = (np.pi * S_selected / domain_size) ** 2

    # callable for basis functions
    def eigen_fun(x):
        return functools.partial(
            _eigen_fnc, L=domain_size / 2, eigen_val=eig_val
        )(x=x - domain_center)

    # calculate spectral density
    spectral_density_fcn = functools.partial(
        _spectral_density_Gaussian, magnitude=scale, lengthscale=lengthscale
    )
    spectral_density = jax.vmap(spectral_density_fcn)(freq=np.sqrt(eig_val))

    return jax.jit(eigen_fun), spectral_density


def _eigen_fnc(x, eigen_val, L):
    return jnp.prod(
        jnp.sqrt(1 / L) * jnp.sin(jnp.sqrt(eigen_val) * (x + L)), axis=1
    )


def _spectral_density_Gaussian(freq, magnitude, lengthscale):
    """
    Calculate the spectral density of the squared exponential kernel with
    individual lengthscales.

    Args:
        freq (ArrayLike): Frequency vector (1D array).
        magnitude (_type_): Variance parameter.
        lengthscale (ArrayLike): Lengthscales for each dimension (1D array).

    Returns:
        float: Spectral density.
    """

    # broadcast to correct shapes
    D = len(freq)
    lengthscale = jnp.broadcast_to(lengthscale, jnp.asarray(freq).shape)

    term1 = magnitude * (2 * jnp.pi) ** (D / 2)
    term2 = jnp.prod(lengthscale)
    exponent = -0.5 * jnp.sum((lengthscale**2) * (freq**2))

    return term1 * term2 * jnp.exp(exponent)
