import matplotlib.pyplot as plt
import numpy as np
import jax

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, LogLocator
import matplotlib
from matplotlib.tri.triangulation import Triangulation



from src.BayesianInferrence import prior_mniw_Predictive
    


plt.rcParams.update({
    "text.usetex": True
})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



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



def apply_basic_formatting(fig, width=8, height=8, font_size=12, dpi=150):
    
    fig.set_size_inches(width*inch_per_cm, height*inch_per_cm)
    
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




def plot_fcn_error_2D(X_in, Mean, X_stats, X_weights, alpha=1.0, norm='log', dpi=150, vmin=1e-4, vmax=3e3):
    
    fig = plt.figure(dpi=dpi)
    gs = fig.add_gridspec(2, 3,  width_ratios=(5, 1, 0.2), height_ratios=(1, 5), hspace=0.05, wspace=0.05)
    
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])
    
    ax_histx.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_histy.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    x_min = np.min(X_in[:,0])
    x_max = np.max(X_in[:,0])
    y_min = np.min(X_in[:,1])
    y_max = np.max(X_in[:,1])
    
    # Triangulation
    triang = Triangulation(X_in[:,0], X_in[:,1])
    
    # Compute the alpha values for each triangle by averaging the alphas of its vertices
    alpha_faces = np.mean(alpha[triang.triangles], axis=1)
    
    
    if norm=='log':
        normalizer = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # plot triangulized mesh
    cntr = ax.tripcolor(
        triang,
        Mean,
        norm=normalizer,
        cmap=imes_colorscale,
        alpha=alpha_faces,
        shading='flat',
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
    fig.colorbar(cntr, cax=cax)
    
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