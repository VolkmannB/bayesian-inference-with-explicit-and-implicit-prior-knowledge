import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from tqdm import tqdm



from src.RGP import ApproximateGP, GaussianRBF
from src.SingleMassOscillator import F_spring, F_damper


def cm2in(cm):
    return cm*0.3937007874



def generate_Animation(X, Y, Sigma_X, H, time, model_para, fps, dpi, width, filne_name='test.mp4'):
    
    # create figure
    width = cm2in(width)
    fig = plt.figure(
        layout='tight', 
        dpi=dpi, 
        figsize=(width, 9/16*width), 
        num=0
        )
    fig.clear()
    
    # fps as integer
    inc = np.ceil((1/fps)/(time[1]-time[0]))
    
    # set up movie writer
    moviewriter = matplotlib.animation.FFMpegWriter(fps=int(1/(time[1]-time[0])/inc))
    
    # helper variables
    F_sd = F_spring(X[:,0], model_para['c1'], model_para['c2']) + F_damper(X[:,1], model_para['d1'], model_para['d2'])
    H_ip = H(H.centers)
       
    # create point grid
    x_plt = np.arange(-5., 5., 0.1)
    dx_plt = np.arange(-5., 5., 0.1)
    grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')
    points = np.dstack([grid_x, grid_y])

    # calculate ground truth for spring damper force
    grid_F = F_spring(grid_x, model_para['c1'], model_para['c2']) + F_damper(grid_y, model_para['d1'], model_para['d2'])
    
    with moviewriter.saving(fig, filne_name, dpi):
        for t in tqdm(range(0, len(time), int(inc)), desc="Generating Plots"):
            
            # get figure0 and clear
            fig = plt.figure(num=0)
            fig.clear()
            
            
            # get GP part of the model
            spring_damper_model = ApproximateGP(
                basis_function=H,
                w0=Sigma_X[t,:,2:].mean(axis=0),
                cov0=np.cov(Sigma_X[t,:,2:].T),
                error_cov=0.001
                )

            # calculate force from GP mapping
            F_sd, pF_sd = spring_damper_model.predict(points[...,np.newaxis,:])
            F_sd = np.squeeze(F_sd)
            pF_sd = np.squeeze(pF_sd)
            
            # error
            e = np.abs(grid_F - F_sd)

            # alpha from confidence
            a = np.abs(1-np.sqrt(pF_sd)/np.max(np.linalg.cholesky(H_ip.T@H_ip*10)))
            
            # plot error
            ax1 = plt.subplot(2,2,1)
            ax1.scatter(H.centers[:,0], H.centers[:,1], marker='.', color='k', label='Inducing Points')
            c = ax1.pcolormesh(grid_x, grid_y, e, label="Ground Truth", vmin=0, vmax=50)
            # ax1.plot(X[:,0], X[:,1], label="State Trajectory", color='r')
            # ax1[0,0].plot(X_pred[:,0], X_pred[:,1], label="State Trajectory", color='r', linestyle='--')
            ax1.scatter(Sigma_X[t,:,0], Sigma_X[t,:,1], color='r', marker='.', label='Samples')
            ax1.set_xlabel("x")
            ax1.set_ylabel("dx")
            ax1.set_xlim(-5., 5.)
            ax1.set_ylim(-5., 5.)
            fig.colorbar(c, ax=ax1)
            
            # write frame
            moviewriter.grab_frame()
            
            
            
def generate_Plott(X, Y, Sigma_X, H, w, Cw, time, model_para, dpi, width):
    
    # get GP part of the model
    spring_damper_model = ApproximateGP(
        basis_function=H,
        w0=w,
        cov0=Cw,
        error_cov=0.001
        )
    
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
    F_sd, pF_sd = spring_damper_model.predict(points[...,np.newaxis,:])
    F_sd = np.squeeze(F_sd)
    pF_sd = np.squeeze(pF_sd)

    # error
    e = np.abs(grid_F - F_sd)

    # aplha from confidence
    a = 1-np.sqrt(pF_sd)/1
    a[a<0]=0
    
    # state trajectory
    X_pred = np.mean(Sigma_X[:,:,:2], axis=1)
    P_pred = np.var(Sigma_X[:,:,:2], axis=1)

    ax1 = plt.subplot(2,2,1)
    ax1.plot(H.centers[:,0], H.centers[:,1], marker='.', color='k', linestyle='none')
    c = ax1.pcolormesh(grid_x, grid_y, e, label="Ground Truth", vmin=0, vmax=10, alpha=a)
    ax1.plot(X_pred[:,0], X_pred[:,1], label="State Trajectory", color='r', linestyle='--')
    ax1.set_xlabel("x")
    ax1.set_ylabel("dx")
    ax1.set_xlim(-5., 5.)
    ax1.set_ylim(-5., 5.)
    fig.colorbar(c, ax=ax1)

    # force
    ax2 = plt.subplot(2,2,2)
    ax2.plot(time, X[:,2], label="F")
    # ax[0,1].plot(time, F_sd, label="F_pred")
    # lower = X_pred[:,2] - np.sqrt(P_pred[:,2])
    # upper = X_pred[:,2] + np.sqrt(P_pred[:,2])
    # ax[0,1].fill_between(time, lower, upper, color='b', alpha=0.2)
    ax2.set_xlabel("time in (s)")
    ax2.set_ylabel("F in (N)")

    # x
    ax3 = plt.subplot(2,2,3)
    ax3.plot(time, Y, 'r.', label="Measurements")
    lower = X_pred[:,0] - np.sqrt(P_pred[:,0])
    upper = X_pred[:,0] + np.sqrt(P_pred[:,0])
    ax3.fill_between(time, lower, upper, color='orange', alpha=0.2)
    ax3.plot(time, X[:,0], label="x", color='b')
    ax3.plot(time, X_pred[:,0], label="x_pred", color='orange', linestyle='--')
    ax3.legend()
    ax3.set_xlabel("time in (s)")
    ax3.set_ylabel("x in (m)")

    # dx
    ax4 = plt.subplot(2,2,4)
    lower = X_pred[:,1] - np.sqrt(P_pred[:,1])
    upper = X_pred[:,1] + np.sqrt(P_pred[:,1])
    ax4.fill_between(time, lower, upper, color='orange', alpha=0.2)
    ax4.plot(time, X[:,1], label="dx", color='b')
    ax4.plot(time, X_pred[:,1], label="dx_pred", color='orange', linestyle='--')
    ax4.set_xlabel("time in (s)")
    ax4.set_ylabel("dx in (m/s)")  