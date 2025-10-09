import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import jax.scipy as jsp

from src.BasisFunctions import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.StateSpaceModel import StateSpaceModel
from src.Algorithm1 import Algorithm1
from src.Algorithm2 import Algorithm2
from src.PGAS import PGAS


#### This section defines the state space model


def f_x(x):
    return 10 * jnp.sinc(x / 7)


def f_y(x):
    return x


#### This section defines the basis function expansion

# basis functions for front and rear tire
N_basis_fcn = 40
lengthscale = 3
basis_fcn, spectral_density = generate_Hilbert_BasisFunction(
    N_basis_fcn,
    np.array([-30, 30]),
    lengthscale,
    50,
)

GP_prior = prior_mniw_2naturalPara(
    np.zeros((1, N_basis_fcn)),
    np.diag(spectral_density),
    np.eye(1),
    10,
)


#### This section defines relevant parameters for the simulation

# simulation parameters
N_particles = 200
N_PGAS_iter = 200
forget_factor = 1.0
t_end = 40.0
time = np.arange(0.0, t_end, 1)
steps = len(time)
key = jax.random.key(12345678)

# initial system state
x0 = np.array([0.0])
P0 = np.diag([1e-4])

# noise
R = np.diag([4])
Q = np.diag([4])

# instantiate state-space model
# in this example, no knowledge about the system dynamics is used
Toy_Example_SSM = StateSpaceModel(
    process_noise=np.zeros((1, 1)),
    output_noise=R,
    transition_model=lambda state, input, *int_var: int_var[0],
    output_model=lambda state, input, *int_var: f_y(int_var[0]),
)

#### This section defines a function for the simulation of the system


def Toy_Example_simulation(key: jax.Array):
    # time series for plot
    X = np.zeros((steps, 1))
    Y = np.zeros((steps, 1))

    # initial value
    X[0, ...] = x0

    # simulation loop
    jit_draw_state = jax.jit(f_x)
    jit_output_mdl = jax.jit(f_y)
    for i in tqdm(range(1, steps), desc="Running System Simulation"):
        key, key_sim = jax.random.split(key)
        X[i] = jit_draw_state(X[i - 1]) + jax.random.normal(key_sim) * jnp.sqrt(
            jnp.squeeze(Q)
        )
        key, key_sim = jax.random.split(key)
        Y[i] = jit_output_mdl(X[i]) + jax.random.normal(key_sim) * jnp.sqrt(
            jnp.squeeze(R)
        )
    return X, Y


# run simulation
key, key_sim = jax.random.split(key)
X, Y = Toy_Example_simulation(key_sim)

# instantiate Algorithm 1
Toy_Example_Algorithm1 = Algorithm1(
    N_samples=N_particles,
    observations=Y,
    inputs=np.zeros((steps, 0)),
    SSM=Toy_Example_SSM,
    forgetting_factor=forget_factor,
    init_state_mean=x0,
    init_state_cov=P0,
    init_int_var_mean=[f_x(x0)],
    init_int_var_cov=[Q],
    GP_prior=[GP_prior],
    basis_fcn=[lambda state, input: basis_fcn(state)],
)

# instantiate Algorithm 2
Toy_Example_Algorithm2 = Algorithm2(
    N_samples=N_particles,
    N_iterations=N_PGAS_iter,
    observations=Y,
    inputs=np.zeros((steps, 0)),
    SSM=Toy_Example_SSM,
    init_state_mean=x0,
    init_state_cov=P0,
    init_int_var_mean=[f_x(x0)],
    init_int_var_cov=[Q],
    GP_prior=[GP_prior],
    basis_fcn=[lambda state, input: basis_fcn(state)],
)

# instantiate PGAS
Toy_Example_PGAS = PGAS(
    N_samples=N_particles,
    N_iterations=N_PGAS_iter * 3,
    observations=Y,
    inputs=np.zeros((steps, 0)),
    init_state_mean=x0,
    init_state_cov=P0,
    likelihood_fcn=lambda obs, state, input: jnp.squeeze(
        jsp.stats.multivariate_normal.logpdf(obs, mean=f_y(state), cov=R)
    ),
    GP_prior=GP_prior,
    basis_fcn=lambda state, input: basis_fcn(state),
)
