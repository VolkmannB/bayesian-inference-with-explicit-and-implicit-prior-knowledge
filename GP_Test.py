import numpy as np
import numpy.typing as npt
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import functools



from src.RGP import gaussian_RBF, prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics, prior_mniw_sampleLikelihood, generate_Hilbert_BasisFunction



def test_function1(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))

def test_function2(x):
    return -x/2 - ((25*x)/(1+x**2)*np.cos(x))


N_phi = 20
inducing_points = np.linspace(-10.0, 10., N_phi)
H = lambda x: gaussian_RBF(
    x,
    inducing_points=inducing_points,
    lengthscale=inducing_points[1] - inducing_points[0]
)
# H, sd = generate_Hilbert_BasisFunction(N_phi, np.array([-10, 10]), 20/N_phi, 10, j_start=1, j_step=1)

MNIW_prior_eta = prior_mniw_2naturalPara(
    np.zeros((2,N_phi)),
    np.eye(N_phi)*100,
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
    phi = H(x)
    
    MNIW_stats = prior_mniw_updateStatistics(*MNIW_stats, np.concatenate([y1, y2]), phi)

MNIW_model = prior_mniw_2naturalPara_inv(
    MNIW_prior_eta[0] + MNIW_stats[0],
    MNIW_prior_eta[1] + MNIW_stats[1],
    MNIW_prior_eta[2] + MNIW_stats[2],
    MNIW_prior_eta[3] + MNIW_stats[3]
)

################################################################################
# sampling test

x = 5
phi = H(x)

pdf_loc = MNIW_model[0] @ phi
pdf_scale = np.linalg.cholesky(MNIW_model[2] * (phi @ MNIW_model[1] @ phi)/(MNIW_model[3]+1))
pdf_1_x = np.linspace(-10,10,500) + pdf_loc[0]
pdf_2_x = np.linspace(-10,10,500) + pdf_loc[1]

pdf_1_y = scipy.stats.t.pdf(x=pdf_1_x, df=MNIW_model[3]+1, loc=pdf_loc[0], scale=pdf_scale[0,0])
pdf_2_y = scipy.stats.t.pdf(x=pdf_2_x, df=MNIW_model[3]+1, loc=pdf_loc[1], scale=pdf_scale[1,1])

keys = jnp.array(jax.random.split(jax.random.key(np.random.randint(100, 1000)), 100000))
samples = jax.vmap(functools.partial(
    prior_mniw_sampleLikelihood,
    M=MNIW_model[0],
    V=MNIW_model[1],
    Psi=MNIW_model[2],
    nu=MNIW_model[3],
    phi=phi
    ))(key=keys)


################################################################################


fig1, ax1 = plt.subplots(2,1,layout='tight')

Xnew = np.linspace(-10.0, 10., 2000)
F1 = test_function1(Xnew)
F2 = test_function2(Xnew)
Phi = jax.vmap(H)(Xnew)

m_BLR = Phi @ MNIW_model[0].T

ax1[0].plot(Xnew, F1, label='True Function')
ax1[0].plot(Xnew, m_BLR[:,0], linestyle='--', color='red', label='GP')
# ax1.fill_between(Xnew, m_BLR - 3*s_BLR, m_BLR + 3*s_BLR, color='red', alpha=0.2)
# ax1.fill_between(Xnew, m_F - 3*s, m_F + 3*s, color='orange', alpha=0.2)
ax1[0].legend()

ax1[1].plot(Xnew, F2, label='True Function')
ax1[1].plot(Xnew, m_BLR[:,1], linestyle='--', color='red', label='GP')


fig2, ax2 = plt.subplots(2,1,layout='tight')

ax2[0].hist(samples[:,0], bins=200, density=True)
ax2[0].plot(pdf_1_x, pdf_1_y)

ax2[1].hist(samples[:,1], bins=200, density=True)
ax2[1].plot(pdf_2_x, pdf_2_y)


plt.show()