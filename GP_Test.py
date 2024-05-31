import numpy as np
import numpy.typing as npt
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics, prior_mniw_Predictive, C2_InversePoly_RBF, bump_RBF



def test_function1(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))

def test_function2(x):
    return -x/2 - ((25*x)/(1+x**2)*np.cos(x))


N_phi = 40
inducing_points = np.linspace(-10.0, 10., N_phi)
basis_fcn = lambda x: C2_InversePoly_RBF(
    x,
    loc=inducing_points,
    lengthscale=inducing_points[1] - inducing_points[0]
)
# H, sd = generate_Hilbert_BasisFunction(N_phi, np.array([-11, 11]), 20/N_phi, 10, j_start=1, j_step=3)

MNIW_prior_eta = prior_mniw_2naturalPara(
    np.zeros((2,N_phi)),
    np.eye(N_phi)*10,
    np.eye(2),
    0
)
MNIW_stats = [
    np.zeros(MNIW_prior_eta[0].shape),
    np.zeros(MNIW_prior_eta[1].shape),
    np.zeros(MNIW_prior_eta[2].shape),
    0
]



rng = np.random.default_rng()
for i in tqdm(range(10000)):
    x = rng.uniform(-10,10,1).flatten()
    y1 = test_function1(x) + rng.normal(0, 1, x.shape)
    y2 = test_function2(x) + rng.normal(0, 1, x.shape)
    phi = basis_fcn(x)
    
    MNIW_stats = list(prior_mniw_updateStatistics(*MNIW_stats, np.concatenate([y1, y2]), phi))
    
    # MNIW_stats[0] *= 0.99
    # MNIW_stats[1] *= 0.99
    # MNIW_stats[2] *= 0.99
    # MNIW_stats[3] *= 0.99



MNIW_model = prior_mniw_2naturalPara_inv(
    MNIW_prior_eta[0] + MNIW_stats[0],
    MNIW_prior_eta[1] + MNIW_stats[1],
    MNIW_prior_eta[2] + MNIW_stats[2],
    MNIW_prior_eta[3] + MNIW_stats[3]
)

################################################################################
# sampling a matrix t distribution

Xnew = np.linspace(-10.0, 10., 2000)
Phi = jax.vmap(basis_fcn)(Xnew)

mean, col_scale, row_scale, df = prior_mniw_Predictive(*MNIW_model, Phi)
std0 = np.diag(col_scale*row_scale[0,0])
std1 = np.diag(col_scale*row_scale[1,1])


################################################################################


fig1, ax1 = plt.subplots(2,1,layout='tight')

F1 = test_function1(Xnew)
F2 = test_function2(Xnew)

ax1[0].plot(Xnew, F1, label='True Function', color='blue')
ax1[0].plot(Xnew, mean[:,0], linestyle='--', color='red', label='GP')
ax1[0].fill_between(Xnew, mean[:,0] - std0, mean[:,0] + std0, color='red', alpha=0.2)
ax1[0].legend()

ax1[1].plot(Xnew, F2, label='True Function', color='blue')
ax1[1].plot(Xnew, mean[:,1], linestyle='--', color='red', label='GP')
ax1[1].fill_between(Xnew, mean[:,1] - std1, mean[:,1] + std1, color='red', alpha=0.2)


plt.show()