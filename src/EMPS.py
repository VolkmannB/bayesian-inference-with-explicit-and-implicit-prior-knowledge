import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.signal
import scipy.io
from tqdm import tqdm

from src.BasisFunctions import generate_Hilbert_BasisFunction
from src.BayesianInferrence import prior_mniw_2naturalPara
from src.StateSpaceModel import StateSpaceModel
from src.Algorithm1 import Algorithm1
from src.Algorithm2 import Algorithm2
from src.PGAS import PGAS


def central_difference_quotient(x, t):
    # Ensure x and t are numpy arrays
    x = np.asarray(x)
    t = np.asarray(t)

    # Initialize dx/dt array
    dxdt = np.zeros_like(x)

    # Compute dt, assuming t is uniformly spaced
    dt = np.diff(t)

    # Forward difference for the first point
    dxdt[0] = (x[1] - x[0]) / dt[0]

    # Central difference for the interior points using vectorized operations
    dxdt[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])

    # Backward difference for the last point
    dxdt[-1] = (x[-1] - x[-2]) / dt[-1]

    return dxdt


#### This section loads the measurment data

# sim para
N_particles = 200
N_PGAS_iter = 800
forget_factor = 0.999
key = jax.random.key(12345678)

### Load data
data = scipy.io.loadmat("src/Measurements/DATA_EMPS.mat")

# calculate reference data
q_ref = data["qm"].flatten()

f_nyq = 500
sos = scipy.signal.butter(4, 100 / f_nyq, btype="lowpass", output="sos")
q_ref = scipy.signal.sosfiltfilt(sos, q_ref)
dq_ref = central_difference_quotient(q_ref, data["t"].flatten())
X = np.vstack([q_ref, dq_ref]).T
X = X[0:-1:10]

# measurements
time = data["t"].flatten()[0:-1:10]
Y = data["qm"].flatten()[0:-1:10]
steps = time.shape[0]
dt = time[1] - time[0]


# initial system state
x0 = np.array([Y[0], 0])
P0 = np.diag([1e-5, 1e-6])
P0_F = np.diag([1e-12])

# process and measurement noise
R = np.diag([1e-4])
Q = np.diag([1e-6, 1e-7])


# input
ctrl_input = (data["vir"] * data["gtau"]).flatten()[0:-1:10]


#### This section defines the basis function expansion

N_basis_fcn = 9
basis_fcn, sd = generate_Hilbert_BasisFunction(
    N_basis_fcn, jnp.array([-0.2, 0.2]), 0.4 / N_basis_fcn, 20
)


def basis_fcn_f(state, input):
    return basis_fcn(state[1])


GP_prior = list(
    prior_mniw_2naturalPara(
        np.zeros((1, N_basis_fcn)), np.diag(sd), np.eye(1) * 4, 2
    )
)

# basis functions for baseline comparison
N_basis_fcn_baseline = N_basis_fcn**3
basis_fcn_baseline, sd_baseline = generate_Hilbert_BasisFunction(
    N_basis_fcn_baseline,
    jnp.array([[-1, 1], [-1, 1], [-1, 1]]),
    0.5 / N_basis_fcn_baseline,
    20,
)


def basis_fcn_f_PGAS(state, input):
    return basis_fcn_baseline(
        jnp.hstack([state, input]) / jnp.array([0.4, 0.4, 160])
    )


GP_prior_PGAS = list(
    prior_mniw_2naturalPara(
        np.zeros((2, N_basis_fcn_baseline)),
        np.diag(sd_baseline),
        np.eye(2),
        2,
    )
)


#### This section defines a function for the simulation of the system


def EMPS_Validation_Simulation(GP_Mean_Alg2, GP_mean_PGAS):
    # Load Validation Data
    data = scipy.io.loadmat("src/Measurements/DATA_EMPS_PULSES.mat")
    time = data["t"].flatten()[0:-1:10]
    Y = data["qm"].flatten()[0:-1:10]
    Tau = (data["vir"] * data["gtau"]).flatten()[0:-1:10]
    steps = time.shape[0]
    dt = time[1] - time[0]

    X_Alg2 = np.zeros((steps, 2))
    X_PGAS = np.zeros((steps, 2))

    X_Alg2[0] = np.array([Y[0], 0])
    X_PGAS[0] = np.array([Y[0], 0])

    for i in tqdm(range(1, steps), desc="Running EMPS Simulation"):
        F = (GP_Mean_Alg2 @ basis_fcn(X_Alg2[i - 1, 1]))[0]
        X_Alg2[i] = f_x(x=X_Alg2[i - 1], tau=Tau[i - 1], F=F, dt=dt)
        X_PGAS[i] = GP_mean_PGAS @ basis_fcn_f_PGAS(X_PGAS[i - 1], Tau[i - 1])

    return np.sqrt(np.mean((X_Alg2[:, 0] - Y) ** 2)), np.sqrt(
        np.mean((X_PGAS[:, 0] - Y) ** 2)
    )


#### This section defines the state space model

# parameters
M = 95.11


# state dynamics
def dx(x, tau, F):
    # x = [q, dq]
    dq = x[1]
    ddq = (tau - F) / M

    return jnp.hstack([dq, ddq])


def dx_linModel(x, tau):
    dq = x[1]
    ddq = (tau - 203.5 * x[1] - 20.39 * jnp.sign(x[1]) + 3.16) / 95.11

    return jnp.hstack([dq, ddq])


# time discrete state space model with Runge-Kutta-4
def f_x(x, tau, F, dt=dt):
    k1 = dx(x, tau, F)
    k2 = dx(x + dt * k1 / 2, tau, F)
    k3 = dx(x + dt * k2 / 2, tau, F)
    k4 = dx(x + dt * k3, tau, F)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@jax.jit
def f_x_linModel(x, tau, dt):
    k1 = dx_linModel(x, tau)
    k2 = dx_linModel(x + dt * k1 / 2, tau)
    k3 = dx_linModel(x + dt * k2 / 2, tau)
    k4 = dx_linModel(x + dt * k3, tau)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# measurement model
def f_y(x):
    return x[0]


# instantiate state-space model
EMPS_SSM = StateSpaceModel(
    process_noise=Q,
    output_noise=R,
    transition_model=lambda state, input, *int_var: f_x(
        state, input, int_var[0], dt
    ),
    output_model=lambda state, input, *int_var: f_y(state),
)


# instantiate Algorithm 1
EMPS_Algorithm1 = Algorithm1(
    N_samples=N_particles,
    observations=Y,
    inputs=ctrl_input,
    SSM=EMPS_SSM,
    forgetting_factor=forget_factor,
    init_state_mean=x0,
    init_state_cov=P0,
    init_int_var_mean=[jnp.array([0])],
    init_int_var_cov=[P0_F],
    GP_prior=[GP_prior],
    basis_fcn=[basis_fcn_f],
)

# instantiate Algorithm 2
EMPS_Algorithm2 = Algorithm2(
    N_samples=N_particles,
    N_iterations=N_PGAS_iter,
    observations=Y,
    inputs=ctrl_input,
    SSM=EMPS_SSM,
    init_state_mean=x0,
    init_state_cov=P0,
    init_int_var_mean=[jnp.array([0])],
    init_int_var_cov=[P0_F],
    GP_prior=[GP_prior],
    basis_fcn=[basis_fcn_f],
)

# instantiate PGAS
EMPS_PGAS_baseline = PGAS(
    N_samples=N_particles,
    N_iterations=N_PGAS_iter * 3,
    observations=Y,
    inputs=ctrl_input,
    init_state_mean=x0,
    init_state_cov=P0,
    likelihood_fcn=lambda obs, state, input: jnp.squeeze(
        jsp.stats.multivariate_normal.logpdf(obs, mean=f_y(state), cov=R)
    ),
    GP_prior=GP_prior_PGAS,
    basis_fcn=basis_fcn_f_PGAS,
)
