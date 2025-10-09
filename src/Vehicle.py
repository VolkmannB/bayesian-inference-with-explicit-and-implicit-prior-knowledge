import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import equinox as eqx

from src.BasisFunctions import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.StateSpaceModel import StateSpaceModel
from src.Algorithm1 import Algorithm1
from src.Algorithm2 import Algorithm2


#### This section defines the state space model

##### default arameters
m = 1720.0
I_zz = 1827.5
l_f = 1.16
l_r = 1.47
g = 9.81
mu_x = 0.9
mu = 0.9
B = 10.0
C = 1.9
E = 0.97


# tire load
def f_Fz(m, l_f, l_r, g):
    l_total = l_f + l_r
    mg = m * g
    F_zf = mg * l_r / l_total
    F_zr = mg * l_f / l_total

    return F_zf, F_zr


# friction MTF curve
def mu_y(alpha, mu=mu, B=B, C=C, E=E):
    return mu * jnp.sin(
        C
        * jnp.arctan(
            B * (1 - E) * jnp.tan(alpha) + E * jnp.arctan(B * jnp.tan(alpha))
        )
    )


# side slip
def f_alpha(x, u, l_f=l_f, l_r=l_r):
    vx_f = u[1]
    vy_f = x[1] + x[0] * l_f

    vx_r = u[1]
    vy_r = x[1] - x[0] * l_r

    return u[0] - jnp.arctan(vy_f / vx_f), -jnp.arctan(vy_r / vx_r)


# state dynamics
def dx(x, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x):
    F_zf, F_zr = f_Fz(m, l_f, l_r, g)

    dv_y = (
        1
        / m
        * (
            F_zf * mu_yf * jnp.cos(u[0])
            + F_zr * mu_yr
            + F_zf * mu_x * jnp.sin(u[0])
        )
        - u[1] * x[0]
    )
    ddpsi = (
        1
        / I_zz
        * (
            l_f * F_zf * mu_yf * jnp.cos(u[0])
            - l_r * F_zr * mu_yr
            + l_f * F_zf * mu_x * jnp.sin(u[0])
        )
    )

    return jnp.hstack([ddpsi, dv_y])


# time discrete state space model with Runge-Kutta-4
def f_x(
    x, u, mu_yf, mu_yr, dt, m=m, I_zz=I_zz, l_f=l_f, l_r=l_r, g=g, mu_x=mu_x
):
    k1 = dx(x, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k2 = dx(x + dt * k1 / 2.0, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k3 = dx(x + dt * k2 / 2.0, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)
    k4 = dx(x + dt * k3, u, mu_yf, mu_yr, m, I_zz, l_f, l_r, g, mu_x)

    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# measurment model
def f_y(
    x,
    u,
    mu_yf,
    mu_yr,
    m=m,
    l_f=l_f,
    l_r=l_r,
    g=g,
    mu_x=mu_x,
    mu=mu,
    B=B,
    C=C,
    E=E,
):
    F_zf, F_zr = f_Fz(m, l_f, l_r, g)

    dv_y = (
        1
        / m
        * (
            F_zf * mu_yf * jnp.cos(u[0])
            + F_zr * mu_yr
            + F_zf * mu_x * jnp.sin(u[0])
        )
        - u[1] * x[0]
    )

    return jnp.tanh(jnp.hstack([x[0], dv_y]))


#### This section defines the basis function expansion

# basis functions for front and rear tire
N_basis_fcn = 20
lengthscale = 2 / 180 * jnp.pi  # 40 /180*jnp.pi / N_basis_fcn
basis_fcn, spectral_density = generate_Hilbert_BasisFunction(
    N_basis_fcn,
    np.array([-30 / 180 * jnp.pi, 30 / 180 * jnp.pi]),
    lengthscale,
    50,
    idx_start=2,
    idx_step=2,
)


def basis_fcn_f(state, input):
    alpha_f, _ = f_alpha(state, input)
    return basis_fcn(alpha_f)


def basis_fcn_r(state, input):
    _, alpha_r = f_alpha(state, input)
    return basis_fcn(alpha_r)


# model prior front tire
GP_prior_f = list(
    prior_mniw_2naturalPara(
        np.zeros((1, N_basis_fcn)),
        np.diag(spectral_density),
        np.eye(1),
        0,
    )
)

# model prior rear tire
GP_prior_r = list(
    prior_mniw_2naturalPara(
        np.zeros((1, N_basis_fcn)),
        np.diag(spectral_density),
        np.eye(1),
        0,
    )
)


#### This section defines relevant parameters for the simulation

# simulation parameters
N_particles = 200
N_PGAS_iter = 800
forget_factor = 0.999
dt = 0.02
t_end = 30.0
time = np.arange(0.0, t_end, dt)
steps = len(time)
key = jax.random.key(12345678)

# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
P0_mu = np.diag([1e-4])

# noise
R = np.diag([0.001 / 180 * np.pi, 1e-3])
Q = np.diag([1e-8, 1e-8])


# control input to the vehicle as [steering angle, longitudinal velocity]
ctrl_input = np.zeros((steps, 2))
ctrl_input[:, 0] = (
    10
    / 180
    * np.pi
    * np.sin(2 * np.pi * time / 5)
    * np.exp(-0.5 * (time - t_end / 2) ** 2 / (t_end / 5) ** 2)
)
ctrl_input[:, 1] = 11.0


# instantiate state-space model
Vehicle_SSM = StateSpaceModel(
    process_noise=Q,
    output_noise=R,
    transition_model=lambda state, input, *int_var: f_x(
        state, input, int_var[0], int_var[1], dt
    ),
    output_model=lambda state, input, *int_var: f_y(
        state, input, int_var[0], int_var[1]
    ),
)

#### This section defines a function for the simulation of the system


def Vehicle_simulation(key: jax.Array):
    # time series for plot
    X = np.zeros((steps, 2))  # sim
    Y = np.zeros((steps, 2))
    mu_f = np.zeros((steps,))
    mu_r = np.zeros((steps,))

    # initial value
    X[0, ...] = x0
    alpha_f, alpha_r = f_alpha(X[0], ctrl_input[0])
    mu_f[0] = mu_y(alpha_f)
    mu_r[0] = mu_y(alpha_r)

    # simulation loop
    jit_draw_state = eqx.filter_jit(Vehicle_SSM.draw_state)
    jit_output_mdl = eqx.filter_jit(Vehicle_SSM.output_mdl)
    for i in tqdm(range(1, steps), desc="Running System Simulation"):
        ####### Model simulation
        key, key_sim = jax.random.split(key)
        X[i] = jit_draw_state(
            key_sim, X[i - 1], ctrl_input[i - 1], mu_f[i - 1], mu_r[i - 1]
        )

        alpha_f, alpha_r = f_alpha(X[i], ctrl_input[i])
        mu_f[i] = mu_y(alpha_f)
        mu_r[i] = mu_y(alpha_r)

        key, key_sim = jax.random.split(key)
        Y[i] = jit_output_mdl(X[i], ctrl_input[i], mu_f[i], mu_r[i])
        Y[i] += jax.random.normal(key_sim, shape=(2,)) * jnp.sqrt(jnp.diag(R))

    return X, Y, mu_f, mu_r


# run simulation
key, key_sim = jax.random.split(key)
X, Y, mu_f, mu_r = Vehicle_simulation(key_sim)

# instantiate Algorithm 1
Vehicle_Algorithm1 = Algorithm1(
    N_samples=N_particles,
    observations=Y,
    inputs=ctrl_input,
    SSM=Vehicle_SSM,
    forgetting_factor=forget_factor,
    init_state_mean=x0,
    init_state_cov=P0,
    init_int_var_mean=[jnp.array([0]), jnp.array([0])],
    init_int_var_cov=[P0_mu, P0_mu],
    GP_prior=[GP_prior_f, GP_prior_r],
    basis_fcn=[basis_fcn_f, basis_fcn_r],
)

# instantiate Algorithm 2
Vehicle_Algorithm2 = Algorithm2(
    N_samples=N_particles,
    N_iterations=N_PGAS_iter,
    observations=Y,
    inputs=ctrl_input,
    SSM=Vehicle_SSM,
    init_state_mean=x0,
    init_state_cov=P0,
    init_int_var_mean=[jnp.array([0]), jnp.array([0])],
    init_int_var_cov=[P0_mu, P0_mu],
    GP_prior=[GP_prior_f, GP_prior_r],
    basis_fcn=[basis_fcn_f, basis_fcn_r],
)
