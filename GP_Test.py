import numpy as np
import numpy.typing as npt
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.BayesianInferrence import prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics, prior_mniw_samplePredictive, generate_Hilbert_BasisFunction, bump_RBF



def test_function1(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))

def test_function2(x):
    return -x/2 - ((25*x)/(1+x**2)*np.cos(x))


N_phi = 40
inducing_points = np.linspace(-10.0, 10., N_phi)
H = lambda x: bump_RBF(
    x,
    inducing_points=inducing_points,
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
for i in tqdm(range(100)):
    x = rng.uniform(-10,10,1).flatten()
    y1 = test_function1(x) + rng.normal(0, 1, x.shape)
    y2 = test_function2(x) + rng.normal(0, 1, x.shape)
    phi = H(x)
    
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
Phi = jax.vmap(H)(Xnew)

V_chol = np.linalg.cholesky(Phi @ MNIW_model[1] @ Phi.T + np.eye(Phi.shape[0]))
Psi_chol = np.linalg.cholesky(MNIW_model[2]/(MNIW_model[3]+1))

N_sample = 1
samples = scipy.stats.t.rvs(df=MNIW_model[3]+1, size=(N_sample, 2, Phi.shape[0]))
f_samples = (MNIW_model[0] @ Phi.T)[None,...] + Psi_chol @ samples @ V_chol.T


################################################################################


fig1, ax1 = plt.subplots(2,1,layout='tight')

F1 = test_function1(Xnew)
F2 = test_function2(Xnew)

m_BLR = Phi @ MNIW_model[0].T

ax1[0].plot(np.repeat(Xnew[:,None], N_sample, axis=-1), f_samples[:,0,:].T, alpha=0.1, color='red')
ax1[0].plot(Xnew, F1, label='True Function', color='blue')
ax1[0].plot(Xnew, m_BLR[:,0], linestyle='--', color='red', label='GP')
# ax1.fill_between(Xnew, m_BLR - 3*s_BLR, m_BLR + 3*s_BLR, color='red', alpha=0.2)
# ax1.fill_between(Xnew, m_F - 3*s, m_F + 3*s, color='orange', alpha=0.2)
ax1[0].legend()

ax1[1].plot(np.repeat(Xnew[:,None], N_sample, axis=-1), f_samples[:,1,:].T, alpha=0.1, color='red')
ax1[1].plot(Xnew, F2, label='True Function')
ax1[1].plot(Xnew, m_BLR[:,1], linestyle='--', color='red', label='GP')


plt.show()