import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm



from src.RGP import ApproximateGP, GaussianRBF, EnsambleGP, gaussian_RBF



def test_function1(x):
    return x/2 + ((25*x)/(1+x**2)*np.cos(x))

def test_function2(x):
    y1 = x[0,:]/2 + ((25*x[0,:])/(1+x[0,:]**2)*np.cos(x[0,:]))
    y2 = x[1,:]/2 - ((25*x[1,:])/(1+x[1,:]**2)*np.cos(x[1,:]))
    y3 = -x[2,:] - ((20*x[2,:])/(1+x[2,:]**2)*np.cos(x[2,:]))
    return np.concatenate((y1[None,...], y2[None,...], y3[None,...]),axis=0)



inducing_points = np.arange(-10.0, 10.1, 0.5)
H = lambda x: gaussian_RBF(
    x,
    inducing_points=inducing_points.reshape((inducing_points.size,1)),
    lengthscale=np.array([0.5])
)

model1 = ApproximateGP(
    basis_function=H,
    n_basis=len(inducing_points)
)
model1._cov *= 100

model3 = ApproximateGP(
    basis_function=H,
    n_basis=len(inducing_points),
    batch_shape=(3,)
)
model3._cov *= 100


model2 = EnsambleGP(
    basis_function=H,
    n_basis=len(inducing_points),
    w0=np.zeros(inducing_points.size),
    cov0=np.eye(inducing_points.size)*10**2,
    N=50,
    error_cov=0.5**2
)


rng = np.random.default_rng()
for i in tqdm(range(1000)):
    X_train = rng.uniform(-10,10,(20,1))
    Y_train = test_function1(X_train) + rng.normal(0, 1, X_train.shape)
    model1.update(X_train, Y_train.flatten(), np.diag(np.ones((X_train.size,))))
    model2.update(X_train, Y_train, np.diag(np.ones((X_train.size,))))
    
    X_train = rng.uniform(-10,10,(3,20,1))
    Y_train = test_function2(X_train) + rng.normal(0, 1, X_train.shape)
    S = np.eye(X_train.shape[1], X_train.shape[1])
    model3.update(X_train, np.squeeze(Y_train), np.concatenate((S[None,...], S[None,...], S[None,...]), axis=0))



################################################################################


fig1, ax1 = plt.subplots(3,1, layout='tight')


Xnew = np.arange(-10,10,0.1)
F = test_function1(Xnew)
m_F, p_F = model1.predict(Xnew.reshape((Xnew.size,1)))
s = np.sqrt(np.diag(p_F))

F_pred = model2.predict(Xnew.reshape((Xnew.size,1)))
F_pred_mean = F_pred.mean(axis=1)

ax1[0].plot(Xnew, F, label='True Function')
ax1[0].plot(Xnew, m_F.flatten(), linestyle='--', color='orange', label='GP mean')
ax1[0].fill_between(Xnew, m_F.flatten() - s, m_F.flatten() + s, color='orange', alpha=0.2)

ax1[1].plot(Xnew, F, label='True Function')
ax1[1].plot(Xnew, F_pred, color='orange', alpha=5/model2.W.shape[0])
ax1[1].plot(Xnew, F_pred_mean.flatten(), linestyle='--', color='orange', label='GP mean')

var = np.var(F_pred, axis=1)
ax1[2].plot(Xnew, F, label='True Function')
ax1[2].plot(Xnew, F_pred_mean.flatten(), linestyle='--', color='orange', label='GP mean')
ax1[2].fill_between(Xnew, F_pred_mean - np.sqrt(var), F_pred_mean + np.sqrt(var), color='orange', alpha=0.2)



fig2, ax2 = plt.subplots(3,1, layout='tight')



Xnew = np.arange(-10,10,0.1)
F = test_function2(np.ones((3,1,1))*Xnew[None,:,None])
m_F, p_F = model3.predict(np.concatenate((Xnew[None,:,None], Xnew[None,:,None], Xnew[None,:,None]), axis=0))

s = np.sqrt(np.diag(p_F[0,...]))
ax2[0].plot(Xnew, F[0,:], label='True Function')
ax2[0].plot(Xnew, m_F[0,:].flatten(), linestyle='--', color='orange', label='GP mean')
ax2[0].fill_between(Xnew, m_F[0,:].flatten() - s, m_F[0,:].flatten() + s, color='orange', alpha=0.2)

s = np.sqrt(np.diag(p_F[1,...]))
ax2[1].plot(Xnew, F[1,:], label='True Function')
ax2[1].plot(Xnew, m_F[1,:].flatten(), linestyle='--', color='orange', label='GP mean')
ax2[1].fill_between(Xnew, m_F[1,:].flatten() - s, m_F[1,:].flatten() + s, color='orange', alpha=0.2)

s = np.sqrt(np.diag(p_F[2,...]))
var = np.var(F_pred, axis=1)
ax2[2].plot(Xnew, F[2,:], label='True Function')
ax2[2].plot(Xnew, m_F[2,:].flatten(), linestyle='--', color='orange', label='GP mean')
ax2[2].fill_between(Xnew, m_F[2,:].flatten() - s, m_F[2,:].flatten() + s, color='orange', alpha=0.2)


plt.show()