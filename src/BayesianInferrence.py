import jax
import jax.numpy as jnp
import jax.scipy as jsc


################################################################################
# Priors


# Solve A X = B for SPD A via Cholesky and triangular solves
def _solve_spd(A: jax.Array, B: jax.Array) -> jax.Array:
    L = jnp.linalg.cholesky(A)
    return jsc.linalg.cho_solve((L, True), B)


# Matrix Normal Inverse Wishart
# For a multivariate Gaussian likelihood with unknown mean and covariance
def prior_mniw_2naturalPara(
    mean: jax.Array, col_cov: jax.Array, row_scale: jax.Array, df: int
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    mean = jnp.atleast_2d(mean)
    row_scale = jnp.atleast_2d(row_scale)

    temp = _solve_spd(col_cov, jnp.hstack([mean.T, jnp.eye(col_cov.shape[0])]))

    eta_0 = temp[:, : mean.shape[0]]
    eta_1 = temp[:, mean.shape[0] :]
    eta_2 = mean @ eta_0 + row_scale

    eta_3 = df

    return eta_0, eta_1, eta_2, eta_3


def prior_mniw_2naturalPara_inv(
    eta_0: jax.Array, eta_1: jax.Array, eta_2: jax.Array, eta_3: float
) -> tuple[jax.Array, jax.Array, jax.Array, float]:
    temp = _solve_spd(eta_1, jnp.hstack([eta_0, jnp.eye(eta_1.shape[0])]))

    mean = temp[:, : eta_0.shape[1]].T
    col_cov = temp[:, eta_0.shape[1] :]
    row_scale = eta_2 - mean @ eta_0
    df = eta_3

    return jnp.atleast_2d(mean), col_cov, jnp.atleast_2d(row_scale), df


def prior_mniw_mean(eta_0: jax.Array, eta_1: jax.Array) -> jax.Array:
    eta_1_sym = 0.5 * (eta_1 + eta_1.T)
    return _solve_spd(eta_1_sym, eta_0).T


def prior_mniw_calcStatistics(
    y: jax.Array, basis: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    T_0 = jnp.outer(basis, y)
    T_1 = jnp.outer(basis, basis)
    T_2 = jnp.outer(y, y)
    T_3 = 1

    return T_0, T_1, T_2, T_3


def prior_mniw_Predictive(
    mean: jax.Array,
    col_cov: jax.Array,
    row_scale: jax.Array,
    df: int,
    basis: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    basis = jnp.atleast_2d(basis)
    col_cov = jnp.atleast_2d(col_cov)
    row_scale = jnp.atleast_2d(row_scale)

    n_b = basis.shape[0]

    # degrees of freedom
    df = df + 1 - row_scale.shape[0]

    # mean
    mean = jnp.squeeze(basis @ mean.T)

    # column variance
    col_scale = basis @ col_cov @ basis.T + jnp.eye(n_b)

    # conditional scale matrix of predictive distribution
    row_scale = row_scale / df

    return mean, col_scale, row_scale, df


def prior_mniw_drawPred(
    key: jax.Array,
    mean: jax.Array,
    col_scale: jax.Array,
    row_scale: jax.Array,
    df: int,
) -> jax.Array:
    # cholesky decomposition
    chol_col_scale = jnp.linalg.cholesky(jnp.atleast_2d(col_scale))
    chol_row_scale = jnp.linalg.cholesky(jnp.atleast_2d(row_scale))

    # draw standard t samples
    samples = jax.random.t(key, df, shape=(chol_row_scale.shape[0]))

    return mean + jnp.squeeze(
        jnp.einsum("ij,j,jk->ik", chol_row_scale, samples, chol_col_scale.T)
    )


def prior_mniw_log_base_measure(T_0, T_1, T_2, T_3):
    n = T_2.shape[0]
    m = T_1.shape[0]

    Psi = T_2 - T_0.T @ _solve_spd(T_1, T_0)
    nu = T_3

    temp_1 = -0.5 * n * m * jnp.log(2 * jnp.pi)
    temp_2 = 0.5 * n * jnp.log(jnp.linalg.det(T_1))
    temp_3 = -0.5 * nu * n * jnp.log(2)
    temp_4 = -jsc.special.multigammaln(nu / 2, n)
    temp_5 = jnp.log(jnp.linalg.det(Psi)) * nu / 2

    return temp_1 + temp_2 + temp_3 + temp_4 + temp_5
