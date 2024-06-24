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
    



def plot_BFE_1D(X_in, Mean, Std, time, X_stats, X_weights):
    
    N_TimeSlices = Mean.shape[0]
    N_tasks = Mean.shape[1]
    
    fig, axes = plt.subplots(N_tasks+1, N_TimeSlices, layout="tight", sharey='row', sharex='col')
    
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



def plot_BFE_2D(X_in, Mean, time, X_stats, X_weights, norm='log'):
    
    N_TimeSlices = Mean.shape[0]
    N_tasks = Mean.shape[1]
    
    fig, axes = plt.subplots(N_tasks+1, N_TimeSlices, layout="tight")
    
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
                norm=normalizer
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
            norm=matplotlib.colors.LogNorm()
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



def plot_Data(Particles, weights, Reference, time):
    
    Particles = np.atleast_3d(Particles)
    Reference = np.atleast_2d(Reference.T).T
    
    N_dim = Particles.shape[-1]
    
    fig, axes = plt.subplots(N_dim, 1, layout='tight', sharex='col')
    axes = np.atleast_1d(axes)
    
    mean = np.einsum('inm,in->im', Particles, weights)
    
    perturbations = Particles - mean[:,None,:]
    std = np.sqrt(np.einsum('inm,in->im', perturbations**2, weights))
    
    for i in range(N_dim):
        
        axes[i].plot(time, Reference[:,i], color=imes_blue)
        axes[i].plot(time, mean[:,i], color=imes_orange, linestyle='--')
        
        axes[i].fill_between(
                time, 
                mean[:,i] - 3*std[:,i], 
                mean[:,i] + 3*std[:,i], 
                facecolor=imes_orange, 
                edgecolor=None, 
                alpha=0.2
                )
        
        # set limits
        axes[i].set_xlim(np.min(time), np.max(time))
    
    axes[i].set_xlabel(r'Time in [$s$]')
        
    return fig, axes



def apply_basic_formatting(fig, width=8, font_size=12, dpi=150):
    
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
    