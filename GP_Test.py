import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm



from src.RGP import gaussian_RBF, update_nig_prior, update_normal_prior



def test_function1(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))

def test_function2(x):
    return -x/2 - ((25*x)/(1+x**2)*np.cos(x))



N_ip = 31
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

BLR_model = [
    np.zeros((N_ip,)),
    np.eye(N_ip)/100,
    0.01,
    0.01
]



rng = np.random.default_rng()
for i in tqdm(range(5000)):
    X_train = rng.uniform(-10,10,1).flatten()
    Y_train = test_function1(X_train) + rng.normal(0, 1, X_train.shape)
    psi = H(X_train)
    
    model = update_normal_prior(*model, psi, Y_train, 1)
    BLR_model = update_nig_prior(*BLR_model, psi, Y_train)



################################################################################


fig1, ax1 = plt.subplots(1,1,layout='tight')


Xnew = np.linspace(-10.0, 10., 2000)
F1 = test_function1(Xnew)
F2 = test_function2(Xnew)
Psi = H(Xnew)
m_F = Psi @ model[0]
# p_F = Psi @ model[1] @ Psi.T
# s = np.diag(np.linalg.cholesky(p_F))

m_BLR = Psi @ BLR_model[0]

ax1.plot(Xnew, m_BLR, linestyle='--', color='red', label='BLR')
# ax1.fill_between(Xnew, m_BLR - 3*s_BLR, m_BLR + 3*s_BLR, color='red', alpha=0.2)
ax1.plot(Xnew, m_F, linestyle='--', color='orange', label='Basis Function Expansion')
# ax1.fill_between(Xnew, m_F - 3*s, m_F + 3*s, color='orange', alpha=0.2)
ax1.plot(Xnew, F1, label='True Function')
ax1.legend()


plt.show()