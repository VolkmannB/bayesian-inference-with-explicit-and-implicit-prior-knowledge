import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt



from src.RGP import ApproximateGP, LimitedSupportRBF



def test_function(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))



inducing_points = np.arange(-10.0, 10.0, 0.5)
H = LimitedSupportRBF(
    centers=inducing_points.reshape((inducing_points.size,1)),
    lengthscale=np.array([0.5]),
    support_radius=3
)

model = ApproximateGP(
    basis_function=H,
    w0=np.zeros(inducing_points.size),
    cov0=np.eye(inducing_points.size)*10,
    error_cov=0.5
)


rng = np.random.default_rng()
X_train = rng.uniform(-10,10,(1,1))
Y_train = test_function(X_train) + rng.normal(0, 1, X_train.shape)
model.update(X_train, Y_train, np.ones((X_train.size,)))



################################################################################


fig, ax = plt.subplots(1,1)


Xnew = np.arange(-10,10,0.1)
F = test_function(Xnew)
m_F, p_F = model.predict(Xnew.reshape((Xnew.size,1)))
s = np.diag(p_F)

ax.plot(Xnew, F, label='True Function')
ax.plot(Xnew, m_F.flatten(), linestyle='--', label='GP mean')
ax.fill_between(Xnew, m_F.flatten() - s, m_F.flatten() + s, color='b', alpha=0.2)



plt.show()