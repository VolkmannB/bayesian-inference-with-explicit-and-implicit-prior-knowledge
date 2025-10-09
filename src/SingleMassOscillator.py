import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm
import equinox as eqx

from src.BasisFunctions import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.StateSpaceModel import StateSpaceModel
from src.Algorithm1 import Algorithm1
from src.Algorithm2 import Algorithm2


#### This section defines the state space model

# parameters
m = 0.2
c1 = 5.0
c2 = 2.0
d1 = 0.4
d2 = 0.4


def F_spring(x):
    return c1 * x + c2 * x**3


def F_damper(dx):
    return d1 * dx * (1 / (1 + d2 * dx * jnp.tanh(dx)))


def dx(x, F, F_sd, m=m):
    return jnp.hstack([x[1], (-F_sd + F) / m])


def f_x(x, F, F_sd, dt):
    # Runge-Kutta 4
    k1 = dx(x, F, F_sd)
    k2 = dx(x + dt / 2.0 * k1, F, F_sd)
    k3 = dx(x + dt / 2.0 * k2, F, F_sd)
    k4 = dx(x + dt * k3, F, F_sd)
    x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


def f_y(x):
    return x[0]


#### This section defines the basis function expansion

# basis functions
N_basis_fcn = 41
basis_fcn, sd = generate_Hilbert_BasisFunction(
    num_fcn=N_basis_fcn,
    domain_boundary=np.array([[-7.5, 7.5], [-7.5, 7.5]]),
    lengthscale=7.5 * 2 / N_basis_fcn,
    scale=100,
)


# parameters of the MNIW prior
GP_prior = prior_mniw_2naturalPara(
    np.zeros((1, N_basis_fcn)),
    np.diag(sd),
    np.eye(1),
    3,
)


#### This section defines relevant parameters for the simulation

# simulation parameters
N_particles = 200
N_PGAS_iter = 800
t_end = 15.0
dt = 0.02
forget_factor = 0.999
time = np.arange(0.0, t_end, dt)
steps = len(time)
key = jax.random.key(12345678)

# initial system state
x0 = np.array([0.0, 0.0])
P0 = np.diag([1e-4, 1e-4])
P0_F = np.diag([1e-12])

# noise
R = np.array([[1e-3]])
Q = np.diag([5e-8, 5e-9])


# external force
F_ext = np.ones((steps,)) * 9.81 * m
F_ext[int(t_end / (3 * dt)) :] = 0
F_ext[int(2 * t_end / (3 * dt)) :] = -9.81 * m


# instantiate state-space model
SMO_SSM = StateSpaceModel(
    process_noise=Q,
    output_noise=R,
    transition_model=lambda state, input, *int_var: f_x(
        state, input, int_var[0], dt
    ),
    output_model=lambda state, input, *int_var: f_y(state),
)


def SingleMassOscillator_simulation(key: jax.Array):
    # time series for plot
    X = np.zeros((steps, 2))  # sim
    Y = np.zeros((steps,))
    F_sd = np.zeros((steps,))

    # set initial value
    X[0, ...] = x0

    # simulation loop
    jit_draw_state = eqx.filter_jit(SMO_SSM.draw_state)
    for i in tqdm(range(1, steps), desc="Running System Simulation"):
        key, key_sim = jax.random.split(key)
        # update system state
        F_sd[i - 1] = F_spring(X[i - 1, 0]) + F_damper(X[i - 1, 1])
        X[i] = jit_draw_state(key_sim, X[i - 1], F_ext[i - 1], F_sd[i - 1])

        # generate measurment
        key, key_sim = jax.random.split(key)
        Y[i] = X[i, 0] + jax.random.normal(key_sim) * jnp.sqrt(jnp.squeeze(R))

    return X, Y, F_sd


# run simulation
key, key_sim = jax.random.split(key)
X, Y, F_sd = SingleMassOscillator_simulation(key_sim)

# instantiate Algorithm 1
SMO_Algorithm1 = Algorithm1(
    N_samples=N_particles,
    observations=Y,
    inputs=F_ext,
    SSM=SMO_SSM,
    forgetting_factor=forget_factor,
    init_state_mean=x0,
    init_state_cov=P0,
    init_int_var_mean=[jnp.array([0])],
    init_int_var_cov=[P0_F],
    GP_prior=[GP_prior],
    basis_fcn=[lambda state, input: basis_fcn(state)],
)

# instantiate Algorithm 2
SMO_Algorithm2 = Algorithm2(
    N_samples=N_particles,
    N_iterations=N_PGAS_iter,
    observations=Y,
    inputs=F_ext,
    SSM=SMO_SSM,
    init_state_mean=x0,
    init_state_cov=P0,
    init_int_var_mean=[jnp.array([0])],
    init_int_var_cov=[P0_F],
    GP_prior=[GP_prior],
    basis_fcn=[lambda state, input: basis_fcn(state)],
)
