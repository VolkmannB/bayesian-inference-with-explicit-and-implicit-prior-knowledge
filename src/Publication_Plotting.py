import matplotlib.pyplot as plt
import numpy as np
import jax

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, LogLocator
import matplotlib



from src.BayesianInferrence import prior_mniw_Predictive
    


plt.rcParams.update({
    "text.usetex": True
})



imes_blue = np.array([0, 80, 155])/255
imes_orange = np.array([231, 123, 41])/255
imes_green = np.array([200, 211, 23])/255

imes_colorscale = matplotlib.colors.LinearSegmentedColormap.from_list(
    "imes_Colorscale",
    [imes_blue, imes_green, imes_orange],
    N=256
    )

aspect_ratio = 16/9
inch_per_cm = 0.3937007874



def generate_BFE_TimeSlices(
    N_slices, 
    X_in, 
    Sigma_X, 
    Sigma_weights,
    Mean, 
    Col_Scale, 
    Row_Scale, 
    DF, 
    basis_fcn, 
    forget_factor
    ):
    
    N_tasks = Row_Scale.shape[-1]
    
    basis = jax.vmap(basis_fcn)(X_in)
    steps = Mean.shape[0]
    
    # allocate variables
    Mean_ = np.zeros((N_slices, N_tasks, basis.shape[0]))
    Std = np.zeros((N_slices, N_tasks, basis.shape[0]))
    X_stats = []
    X_weights = []
    Time = np.zeros(N_slices)
    
    for i in range(N_slices):
        
        # index of time slice
        step = int((i+1)/N_slices*steps)-1
        
        # calculate predictive distribution
        mean, col_scale, row_scale, df = prior_mniw_Predictive(
            mean=Mean[step], 
            col_cov=Col_Scale[step], 
            row_scale=Row_Scale[step], 
            df=DF[step], 
            basis=basis
            )
        Mean_[i] = mean.T
        
        # calculate standard deviation
        for j in range(N_tasks):
            Std[i,j] = np.sqrt(np.diag(col_scale*row_scale[j,j]))
        
        # window = int(np.maximum(0, 1/(1-forget_factor)))
        # window = int(np.maximum(0, step - df))
        weights = np.ones(Sigma_weights[:step].shape) * forget_factor
        weights[-1,:] = 1
        weights = np.cumprod(weights.T, axis=-1).T
        weights = weights * Sigma_weights[:step]
        X_weights.append(weights)
        
        X_stats.append(Sigma_X[:step])
        
        Time[i] = step
        
    
    
    return Mean_, Std, X_stats, X_weights, Time
    



def plot_BFE_1D(X_in, Mean, Std, time, X_stats, X_weights, dpi=150):
    
    N_TimeSlices = Mean.shape[0]
    N_tasks = Mean.shape[1]
    
    fig, axes = plt.subplots(
        N_tasks+1, 
        N_TimeSlices, 
        layout="tight", 
        sharey='row', 
        sharex='col', 
        dpi=dpi
        )
    
    x_min = np.min(X_in)
    x_max = np.max(X_in)
    
    for i in range(N_TimeSlices):
        
        # plot slice for each task
        for j in range(N_tasks):
            axes[j,i].plot(X_in, Mean[i,j], color=imes_blue)
            axes[j,i].fill_between(
                X_in, 
                Mean[i,j] - 3*Std[i,j], 
                Mean[i,j] + 3*Std[i,j], 
                facecolor=imes_blue, 
                edgecolor=None, 
                alpha=0.2
                )
            axes[j,i].set_xlim(x_min, x_max)
            
        
        # set timestamp of slice as title
        axes[0,i].set_title(f"$t={np.round(time[i],2)} s$")
        
        # plot histogram of input data
        axes[N_tasks,i].hist(
            X_stats[i].flatten(), 
            bins=50, 
            range=(x_min, x_max), 
            weights=X_weights[i].flatten(), 
            color=imes_blue,
            log=True
            )
        axes[N_tasks,i].set_xlim(x_min, x_max)
    
    for ax in axes.flat:
        ax.grid(which='major', color='gray', alpha=0.2)
        
        
    
    return fig, axes



def plot_BFE_2D(X_in, Mean, time, X_stats, X_weights, norm='log', dpi=150):
    
    N_TimeSlices = Mean.shape[0]
    N_tasks = Mean.shape[1]
    
    fig, axes = plt.subplots(N_tasks+1, N_TimeSlices, layout="tight", dpi=dpi)
    
    x_min = np.min(X_in[:,0])
    x_max = np.max(X_in[:,0])
    y_min = np.min(X_in[:,1])
    y_max = np.max(X_in[:,1])
    z_min = np.min(Mean)
    z_max = np.max(Mean)
    
    if norm=='log':
        normalizer = matplotlib.colors.LogNorm()
    else:
        normalizer = matplotlib.colors.Normalize()
    
    for i in range(N_TimeSlices):
        
        # plot slice for each task
        for j in range(N_tasks):
            cntr = axes[j,i].tripcolor(
                X_in[:,0], 
                X_in[:,1], 
                Mean[i,j],
                norm=normalizer,
                cmap=imes_colorscale
                )
            axes[j,i].set_xlim(x_min, x_max)
            axes[j,i].set_ylim(y_min, y_max)
    
            # add colorbar for tasks
            if i == N_TimeSlices-1:
                fig.colorbar(cntr, ax=axes[j,i])
        
        # set timestamp of slice as title
        axes[0,i].set_title(f"$t={np.round(time[i],2)} s$")
        
        # plot histogram of input data
        hist = axes[N_tasks,i].hist2d(
            x=X_stats[i][...,0].flatten(),
            y=X_stats[i][...,1].flatten(),
            bins=50, 
            range=((x_min, x_max), (y_min, y_max)), 
            weights=X_weights[i].flatten(), 
            norm=matplotlib.colors.LogNorm(),
            cmap=imes_colorscale
            )
        axes[N_tasks,i].set_xlim(x_min, x_max)
        axes[N_tasks,i].set_ylim(y_min, y_max)
        
    # add colorbar for histogram
    fig.colorbar(hist[3], ax=axes[N_tasks,i])
    
    for ax in axes.flat:
        ax.grid(which='major', color='gray', alpha=0.2)
        
        
    
    return fig, axes



def set_font_size(fig, size):
    for ax in fig.get_axes():
        ax.title.set_fontsize(size)
        ax.xaxis.label.set_fontsize(size)
        ax.yaxis.label.set_fontsize(size)
        if isinstance(ax,Axes3D):
            ax.zaxis.label.set_fontsize(size)
        ax.tick_params(axis='both', which='major', labelsize=size)
        legend = ax.get_legend()
        if legend:
            plt.setp(legend.get_texts(), fontsize=size)
    for text in fig.findobj(match=plt.Text):
        text.set_fontsize(size)



def plot_Data(Particles, weights, Reference, time, dpi=150):
    
    Particles = np.atleast_3d(Particles)
    Reference = np.atleast_2d(Reference.T).T
    
    N_dim = Particles.shape[-1]
    
    fig, axes = plt.subplots(N_dim, 1, layout='tight', sharex='col', dpi=dpi)
    axes = np.atleast_1d(axes)
    
    mean = np.einsum('inm,in->im', Particles, weights)
    
    perturbations = Particles - mean[:,None,:]
    std = np.sqrt(np.einsum('inm,in->im', perturbations**2, weights))
    
    for i in range(N_dim):
        
        axes[i].plot(time, mean[:,i], color=imes_blue)
        axes[i].plot(time, Reference[:,i], color='red', linestyle='--')
        
        axes[i].fill_between(
                time, 
                mean[:,i] - 3*std[:,i], 
                mean[:,i] + 3*std[:,i], 
                facecolor=imes_blue, 
                edgecolor=None, 
                alpha=0.2
                )
        
        # set limits
        axes[i].set_xlim(np.min(time), np.max(time))
        
    return fig, axes



def apply_basic_formatting(fig, width=8, aspect_ratio=aspect_ratio, font_size=12, dpi=150):
    
    fig.set_size_inches(width*inch_per_cm, width*inch_per_cm/aspect_ratio)
    
    set_font_size(fig, font_size)
    
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.useMathText = True
    
    for ax in fig.get_axes():
        ax.grid(which='major', color='gray', alpha=0.2)
        # ax.xaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(formatter)
        # if isinstance(ax,Axes3D):
        #     ax.zaxis.set_major_formatter(formatter)
        
    fig.set_dpi(dpi)



def plot_PGAS_iterrations(Trajectories, time, iter_idx, dpi=150):
    
    Trajectories = np.atleast_3d(Trajectories)
    
    N_Slices = iter_idx.shape[0]
    N_tasks = Trajectories.shape[-1]
    
    fig, axes = plt.subplots(
        N_tasks, 
        N_Slices, 
        layout="tight", 
        sharey='row', 
        sharex='col', 
        dpi=dpi
        )
    axes = np.atleast_2d(axes)
    
    
    for i in range(N_Slices):
        
        for j in range(N_tasks):
            
            # plot reference
            axes[j,i].plot(time, Trajectories[:,int(iter_idx[i]-1),j], 
                        color=imes_blue,
                        label='Ref')
            
            # plot drawn sample
            axes[j,i].plot(time, Trajectories[:,int(iter_idx[i]),j], 
                        color=imes_orange,
                        linestyle='',
                        marker='.', 
                        markersize=3,
                        label='sample'
                        )
            
        axes[0,i].set_title(f"Iteration ${int(iter_idx[i])}$")
        axes[-1,i].set_xlabel(r'Time in [$s$]')
        axes[-1,i].set_xlim(np.min(time), np.max(time))
    
    axes[0,-1].legend()
        
    return fig, axes



def plot_fcn_error_2D(X_in, Mean, X_stats, X_weights, alpha=1.0, norm='log', dpi=150):
    
    fig = plt.figure(dpi=dpi)
    gs = fig.add_gridspec(2, 2,  width_ratios=(5, 1), height_ratios=(1, 5), hspace=0.05, wspace=0.05)
    
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    ax_histx.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_histy.tick_params(axis='y', which='both', left=False, right=False, labelbottom=False)
    
    x_min = np.min(X_in[:,0])
    x_max = np.max(X_in[:,0])
    y_min = np.min(X_in[:,1])
    y_max = np.max(X_in[:,1])
    
    
    if norm=='log':
        normalizer = matplotlib.colors.LogNorm()
    else:
        normalizer = matplotlib.colors.Normalize()
    
    # plot triangulized mesh
    cntr = ax.tripcolor(
        X_in[:,0], 
        X_in[:,1], 
        Mean,
        norm=normalizer,
        cmap=imes_colorscale,
        alpha=alpha,
        shading='gouraud',
        edgecolors='none'
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # plot histogram
    ax_histx.hist(
        X_stats[...,0].flatten(), 
        bins=np.linspace(x_min, x_max, 100),
        weights=X_weights.flatten(), 
        color=imes_blue,
        log=False)
    ax_histy.hist(
        X_stats[...,1].flatten(), 
        bins=np.linspace(x_min, x_max, 100), 
        weights=X_weights.flatten(), 
        color=imes_blue,
        log=False,
        orientation='horizontal',)
    
    # add colorbar
    fig.colorbar(cntr, ax=ax_histy)
    
    return fig, [ax, ax_histx, ax_histy]



def plot_fcn_error_1D(X_in, Mean, Std, X_stats, X_weights, dpi=150):
    
    Mean = np.atleast_2d(Mean)
    Std = np.atleast_2d(Std)
    N_tasks = Mean.shape[0]
    
    x_min = np.min(X_in)
    x_max = np.max(X_in)
    
    fig = plt.figure(dpi=dpi)
    gs = fig.add_gridspec(
        N_tasks+1, 1, 
        height_ratios=(1, *(5*np.ones((N_tasks,)))), 
        hspace=0.05, 
        wspace=0.05)
    
    ax = []
    for i in range(0,N_tasks):
        
        if i == 0:
            ax.append(fig.add_subplot(gs[i+1, 0]))
        else:
            ax.append(fig.add_subplot(gs[i+1, 0], sharex=ax[i-1]))
    
        # plot function
        ax[i].plot(X_in, Mean[i], color=imes_blue)
        ax[i].fill_between(
            X_in, 
            Mean[i] - 3*Std[i], 
            Mean[i] + 3*Std[i], 
            color=imes_blue, 
            alpha=0.2)
        ax[i].set_xlim(x_min, x_max)
    
    # plot histogram
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax[-1])
    ax_histx.hist(
        X_stats.flatten(), 
        bins=np.linspace(x_min, x_max, 100),
        weights=X_weights.flatten(), 
        color=imes_blue)
    ax_histx.tick_params(
        axis='x', 
        which='both', 
        bottom=False, 
        top=False, 
        labelbottom=False)
    
    return fig, [ax, ax_histx]