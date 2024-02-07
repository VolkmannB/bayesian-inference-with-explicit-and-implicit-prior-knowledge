import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm



from src.RGP import BasisFunctionExpansion, gaussian_RBF, LinearBayesianRegression



def test_function1(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))

def test_function2(x):
    return -x/2 - ((25*x)/(1+x**2)*np.cos(x))



inducing_points = np.arange(-10.0, 10.1, 0.5)
H = lambda x: gaussian_RBF(
    x,
    inducing_points=inducing_points.reshape((inducing_points.size,1)),
    lengthscale=np.array([0.5])
)

class Model(BasisFunctionExpansion):
    
    def __init__(self) -> None:
        super().__init__(inducing_points.size)
        
        self.register_basis_function(H)
        self.initialize_prior(
            np.zeros((inducing_points.size,)),
            np.eye(inducing_points.size)*100
            )
        
model = Model()


class BLR(LinearBayesianRegression):
    
    def __init__(self) -> None:
        super().__init__(inducing_points.size)
        
        self.register_basis_function(H)
        self.initialize_prior(
            np.zeros((inducing_points.size,)),
            np.eye(inducing_points.size)*100
            )
        
BLR_model = BLR()



rng = np.random.default_rng()
for i in tqdm(range(5000)):
    X_train = rng.uniform(-10,10,1)
    Y_train = test_function1(X_train) + rng.normal(0, 1, X_train.shape)
    model.update(X_train, Y_train, 1)
    BLR_model.update(X_train, Y_train)



################################################################################


fig1, ax1 = plt.subplots(1,1,layout='tight')


Xnew = np.arange(-10,10,0.1)
F1 = test_function1(Xnew)
F2 = test_function2(Xnew)
m_F, p_F = model(Xnew.reshape((Xnew.size,1)))
s = np.sqrt(np.diag(p_F))

m_BLR = BLR_model(Xnew)
n_sample=100
m_BLR_sample = BLR_model.sample_f(Xnew, n_sample)
s_BLR = np.std(m_BLR_sample, axis=0)

ax1.plot(Xnew, m_BLR, linestyle='--', color='red', label='BLR')
ax1.fill_between(Xnew, m_BLR - 3*s_BLR, m_BLR + 3*s_BLR, color='red', alpha=0.2)
ax1.plot(Xnew, m_F, linestyle='--', color='orange', label='Basis Function Expansion')
ax1.fill_between(Xnew, m_F - 3*s, m_F + 3*s, color='orange', alpha=0.2)
ax1.plot(Xnew, F1, label='True Function')
ax1.legend()


plt.show()