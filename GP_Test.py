import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm



from src.RGP import gaussian_RBF, update_normal_prior, prior_mniw_2naturalPara, prior_mniw_2naturalPara_inv, prior_mniw_updateStatistics



def test_function1(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))

def test_function2(x):
    return -x/2 - ((25*x)/(1+x**2)*np.cos(x))



N_ip = 40
inducing_points = np.linspace(-10.0, 10., N_ip)
H = lambda x: gaussian_RBF(
    x,
    inducing_points=inducing_points,
    lengthscale=np.array([0.5])
)

        
model = [
    np.zeros((N_ip,)),
    np.eye(N_ip)*100
]

MNIW_prior_eta = prior_mniw_2naturalPara(
    np.zeros((2,N_ip)),
    np.eye(N_ip)*100,
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
    psi = H(x)
    
    MNIW_stats = prior_mniw_updateStatistics(*MNIW_stats, np.concatenate([y1, y2]), psi)

MNIW_model = prior_mniw_2naturalPara_inv(
    MNIW_prior_eta[0] + MNIW_stats[0],
    MNIW_prior_eta[1] + MNIW_stats[1],
    MNIW_prior_eta[2] + MNIW_stats[2],
    MNIW_prior_eta[3] + MNIW_stats[3]
)



################################################################################


fig1, ax1 = plt.subplots(2,1,layout='tight')


Xnew = np.linspace(-10.0, 10., 2000)
F1 = test_function1(Xnew)
F2 = test_function2(Xnew)
Psi = H(Xnew)

m_BLR = Psi @ MNIW_model[0].T

ax1[0].plot(Xnew, F1, label='True Function')
ax1[0].plot(Xnew, m_BLR[:,0], linestyle='--', color='red', label='GP')
# ax1.fill_between(Xnew, m_BLR - 3*s_BLR, m_BLR + 3*s_BLR, color='red', alpha=0.2)
# ax1.fill_between(Xnew, m_F - 3*s, m_F + 3*s, color='orange', alpha=0.2)
ax1[0].legend()

ax1[1].plot(Xnew, F2, label='True Function')
ax1[1].plot(Xnew, m_BLR[:,1], linestyle='--', color='red', label='GP')


plt.show()