import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm



from src.RGP import BasisFunctionExpansion, gaussian_RBF



def test_function1(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))

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



rng = np.random.default_rng()
for i in tqdm(range(10000)):
    X_train = rng.uniform(-10,10,1)
    Y_train = test_function1(X_train) + rng.normal(0, 1, X_train.shape)
    model.fit_BOLS(X_train, Y_train, [1])



################################################################################


fig1, ax1 = plt.subplots(1,1,layout='tight')


Xnew = np.arange(-10,10,0.1)
F = test_function1(Xnew)
m_F, p_F = model(Xnew.reshape((Xnew.size,1)))
s = np.sqrt(np.diag(p_F))

ax1.plot(Xnew, F, label='True Function')
ax1.plot(Xnew, m_F.flatten(), linestyle='--', color='orange', label='GP mean')
ax1.fill_between(Xnew, m_F.flatten() - 3*s, m_F.flatten() + 3*s, color='orange', alpha=0.2)


plt.show()