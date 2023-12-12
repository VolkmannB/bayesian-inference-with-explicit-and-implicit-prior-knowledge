import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from tqdm import tqdm



from src.RGP import ApproximateGP, GaussianRBF
from src.SingleMassOscillator import F_spring, F_damper


def cm2in(cm):
    return cm*0.3937007874



def generate_Animation(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, W, CW, time, model_para, dpi, width, fps, filne_name='test.mp4'):
    
    # create figure
    width = cm2in(width)
    fig = plt.figure(
        layout='tight', 
        dpi=dpi, 
        figsize=(width, 9/16*width), 
        num=0
        )
    fig.clear()
    
    # fps to step increments if time axis
    inc = np.ceil((1/fps)/(time[1]-time[0]))
    
    # set up movie writer
    moviewriter = matplotlib.animation.FFMpegWriter(fps=int(1/(time[1]-time[0])/inc))
    
    # helper variables
    H_ip = H(H.centers)
       
    # create point grid
    x_plt = np.arange(-5., 5., 0.1)
    dx_plt = np.arange(-5., 5., 0.1)
    grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')
    points = np.dstack([grid_x, grid_y])
    
    # state trajectory as gaussian
    X_pred = np.mean(Sigma_X, axis=1)
    P_pred = np.var(Sigma_X, axis=1)

    # calculate ground truth for spring damper force
    grid_F = F_spring(grid_x, model_para['c1'], model_para['c2']) + F_damper(grid_y, model_para['d1'], model_para['d2'])
    
    with moviewriter.saving(fig, filne_name, dpi):
        for t in tqdm(range(0, len(time), int(inc)), desc="Generating Plots"):
            
            # get figure0 and clear
            fig = plt.figure(num=0)
            fig.clear()
            
            
            # get GP part of the model
            spring_damper_model = ApproximateGP(
                basis_function=H
                )
            spring_damper_model._mean = W[t,:]
            spring_damper_model._cov = CW[t,...]

            # calculate force from GP mapping
            GPF_sd, GPpF_sd = spring_damper_model.predict(points[...,np.newaxis,:])
            GPF_sd = np.squeeze(GPF_sd)
            GPpF_sd = np.squeeze(GPpF_sd)
            
            # error
            e = grid_F - GPF_sd

            # alpha from confidence
            a = 1-np.sqrt(GPpF_sd)/np.sqrt(np.max(CW[0,...]))
            a[a<0]=0
            # a=1
            
            # plot error
            ax1 = plt.subplot(2,2,1)
            draw_GPerror(ax1, grid_x, grid_y, e, a)
            draw_Ensamble(ax1, Sigma_X[t,:,0], Sigma_X[t,:,1])

            # force
            ax2 = plt.subplot(2,2,2)
            draw_F(ax2, F_sd[:t], F_pred[:t], PF_pred[:t], time[:t], time[-1])

            # x
            ax3 = plt.subplot(2,2,3)
            draw_x(ax3, X[:t,0], X_pred[:t,0], P_pred[:t,0], time[:t], time[-1])

            # dx
            ax4 = plt.subplot(2,2,4)
            draw_dx(ax4, X[:t,1], X_pred[:t,1], P_pred[:t,1], time[:t], time[-1])
            
            # write frame
            moviewriter.grab_frame()
            
            
            
def generate_Plot(X, Y, F_sd, Sigma_X, F_pred, PF_pred, H, w, Cw, time, model_para, dpi, width):
    
    # get GP part of the model
    spring_damper_model = ApproximateGP(
        basis_function=H
        )
    spring_damper_model._mean = w
    spring_damper_model._cov = Cw
    
    # create figure
    width = cm2in(width)
    fig = plt.figure(
        layout='tight', 
        dpi=dpi, 
        figsize=(width, 9/16*width), 
        num=1
        )
    fig.clear()

    # create point grid
    x_plt = np.arange(-5., 5., 0.1)
    dx_plt = np.arange(-5., 5., 0.1)
    grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')

    # calculate ground truth for spring damper force
    grid_F = F_spring(grid_x, model_para['c1'], model_para['c2']) + F_damper(grid_y, model_para['d1'], model_para['d2'])
    
    # calculate force from GP mapping
    points = np.dstack([grid_x, grid_y])
    GPF_sd, GPpF_sd = spring_damper_model.predict(points[...,np.newaxis,:])
    GPF_sd = np.squeeze(GPF_sd)
    GPpF_sd = np.squeeze(GPpF_sd)

    # GP error
    e = grid_F - GPF_sd

    # aplha from confidence
    a = 1-np.sqrt(GPpF_sd)/np.sqrt(np.max(Cw))
    a[a<0]=0
    
    # state trajectory
    X_pred = np.mean(Sigma_X, axis=1)
    P_pred = np.var(Sigma_X, axis=1)

    # spring damper field
    ax1 = plt.subplot(2,2,1)
    draw_GPerror(ax1, grid_x, grid_y, e, a)

    # force
    ax2 = plt.subplot(2,2,2)
    draw_F(ax2, F_sd, F_pred, PF_pred, time, time[-1])

    # x
    ax3 = plt.subplot(2,2,3)
    draw_x(ax3, X[:,0], X_pred[:,0], P_pred[:,0], time, time[-1])

    # dx
    ax4 = plt.subplot(2,2,4)
    draw_dx(ax4, X[:,1], X_pred[:,1], P_pred[:,1], time, time[-1])



def draw_GPerror(
    ax: plt.Axes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    error: np.ndarray,
    alpha: np.ndarray
    ):
    
    c = ax.pcolormesh(grid_x, grid_y, error, label="Ground Truth", vmin=-10, vmax=10, alpha=alpha)
    plt.colorbar(c, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("dx")
    ax.set_xlim(-5., 5.)
    ax.set_ylim(-5., 5.)



def draw_x(
    ax: plt.Axes,
    X: np.ndarray,
    X_pred: np.ndarray,
    P_pred: np.ndarray,
    t: np.ndarray,
    t_end: float
    ):
    
    lower = X_pred - np.sqrt(P_pred)
    upper = X_pred + np.sqrt(P_pred)
    ax.fill_between(t, lower, upper, color='orange', alpha=0.2)
    ax.plot(t, X, label="x", color='royalblue')
    ax.plot(t, X_pred, label="x_pred", color='orange', linestyle='--')
    ax.legend()
    ax.set_xlim(0, t_end)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("time in (s)")
    ax.set_ylabel("x in (m)")



def draw_dx(
    ax: plt.Axes,
    X: np.ndarray,
    X_pred: np.ndarray,
    P_pred: np.ndarray,
    t: np.ndarray,
    t_end: float
    ):
    
    lower = X_pred - np.sqrt(P_pred)
    upper = X_pred + np.sqrt(P_pred)
    ax.fill_between(t, lower, upper, color='orange', alpha=0.2)
    ax.plot(t, X, label="dx", color='royalblue')
    ax.plot(t, X_pred, label="dx_pred", color='orange', linestyle='--')
    ax.legend()
    ax.set_xlim(0, t_end)
    ax.set_ylim(-7, 7)
    ax.set_xlabel("time in (s)")
    ax.set_ylabel("dx in (m/s)")



def draw_F(
    ax: plt.Axes,
    F: np.ndarray,
    F_pred: np.ndarray,
    P_pred: np.ndarray,
    t: np.ndarray,
    t_end: float
    ):
    
    lower = F_pred - np.sqrt(P_pred)
    upper = F_pred + np.sqrt(P_pred)
    ax.fill_between(t, lower, upper, color='orange', alpha=0.2)
    ax.plot(t, F, label="F", color='royalblue')
    ax.plot(t, F_pred, label="F_pred", color='orange', linestyle='--')
    ax.legend()
    ax.set_xlim(0, t_end)
    ax.set_ylim(-50, 60)
    ax.set_xlabel("time in (s)")
    ax.set_ylabel("F in (N)")



def draw_Ensamble(ax: plt.Axes, x, dx):
    
    ax.scatter(x, dx, marker='.', color='r')