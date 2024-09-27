import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import matplotlib.pyplot as plt
from src.Publication_Plotting import plot_Data, apply_basic_formatting
from src.Publication_Plotting import plot_fcn_error_1D, imes_blue, imes_orange, imes_green
from src.Publication_Plotting import plot_BFE_2D, generate_BFE_TimeSlices, plot_fcn_error_2D

from src.BayesianInferrence import prior_mniw_Predictive, prior_mniw_2naturalPara_inv



fig_overall, axs_overall = plt.subplots(3, 3, gridspec_kw={'height_ratios': [1.5, 1, 1]}, layout='tight', sharex='col')

fontsize_overall = 8
fontsize_miniplots = 8
figsize_miniplots = 4 #cm
dpi_miniplots = 300

################################################################################
################################################################################
############################# SingleMassOscillator #############################
################################################################################
################################################################################

from src.SingleMassOscillator import F_spring, F_damper, basis_fcn, GP_model_prior

################################################################################
# Loading

# Load the arrays from the file
loaded = np.load('SingleMassOscillator_APF_saved.npz')

# Access individual arrays by their names
# Simulation
X = loaded['X']
Y = loaded['Y']
F_sd = loaded['F_sd']

# Particle filter
Sigma_X = loaded['Sigma_X']
Sigma_F = loaded['Sigma_F']
weights = loaded['weights']
Mean_F = loaded['Mean_F']
Col_cov_F = loaded['Col_cov_F']
Row_scale_F = loaded['Row_scale_F']
df_F = loaded['df_F']
fcn_var=loaded['fcn_var']
fcn_mean=loaded['fcn_mean']
# alpha=alpha,
time=loaded['time']
steps=loaded['steps']


################################################################################
# Plots





### plot time slices of the learned spring-damper function
x_plt = np.linspace(-5., 5., 50)
dx_plt = np.linspace(-5., 5., 50)
grid_x, grid_y = np.meshgrid(x_plt, dx_plt, indexing='xy')
X_in = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
basis_in = jax.vmap(basis_fcn)(X_in)

N_slices = 4
index2 = (np.array(range(N_slices))+1)/N_slices*(steps-1)
# index = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])*(steps-1)
index = np.array([0.05, 0.3, 0.8])*(steps-1)


# true spring damper force
F_sd_true = jax.vmap(F_spring)(X_in[:,0]) + jax.vmap(F_damper)(X_in[:,1])

# function values with GP prior
GP_prior = prior_mniw_2naturalPara_inv(
            GP_model_prior[0],
            GP_model_prior[1],
            GP_model_prior[2],
            GP_model_prior[3]
        )
_, col_scale_prior, row_scale_prior, _ = prior_mniw_Predictive(
    mean=GP_prior[0], 
    col_cov=GP_prior[1], 
    row_scale=GP_prior[2], 
    df=GP_prior[3], 
    basis=basis_in)
fcn_var_prior = np.diag(col_scale_prior-1) * row_scale_prior[0,0]
del col_scale_prior, row_scale_prior, GP_prior

# normalize variance to create transparency effect
fcn_alpha = np.maximum(np.minimum(1 - fcn_var/fcn_var_prior, 1), 0)

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_2D(
        X_in, 
        Mean=np.abs(fcn_mean[int(i)]-F_sd_true), 
        X_stats=Sigma_X[:int(i)], 
        X_weights=weights[:int(i)], 
        alpha=fcn_alpha[int(i)],
        max_x=250, max_y=100)
    # ax_fcn_e[0].set_xlabel(r"$s$ $\mathrm{in}$ $\mathrm{m}$")
    # ax_fcn_e[0].set_ylabel(r"$\dot{s}$ $\mathrm{in}$ $\mathrm{m/s}$")

    ax_fcn_e[0].set_yticks([-4,0,4],['$-4$',r'$s$','$4$'])
    ax_fcn_e[0].set_xticks([-4,0,4],['$-4$',r'$\dot{s}$','$4$'])
    ax_fcn_e[1].text(5.2,75,r'$\# \mathrm{Data}$')
    # ax_fcn_e[2].set_yticklabels([])

    apply_basic_formatting(fig_fcn_e, width=figsize_miniplots, height=1, font_size=fontsize_miniplots)
    # fig_fcn_e.tight_layout()
    fig_fcn_e.savefig(f"SingleMassOscillator_APF_Fsd_fcn_{np.round(time[int(i)],3)}.png",dpi=dpi_miniplots, bbox_inches='tight')




# plot weighted RMSE of GP over entire function space
fcn_var = fcn_var + 1e-4 # to avoid dividing by zero
v1 = np.sum(1/fcn_var, axis=-1)
wRMSE = np.sqrt(v1/(v1**2 - v1) * jnp.sum((fcn_mean - F_sd_true) ** 2 / fcn_var, axis=-1))
axs_overall[0,0].plot(
    time,
    wRMSE,
    color=imes_blue
)
axs_overall[0,0].set_ylabel(r"$\mathrm{wRMSE}$ $\mathrm{in}$ $\mathrm{N}$")
axs_overall[0,0].set_ylim(0)

for i in index:
    axs_overall[0,0].plot([time[int(i)], time[int(i)]], [0, wRMSE[int(i)]*1.5], color="black", linewidth=0.8)



# plot the state estimations
Particles=np.concatenate([Sigma_X, Sigma_F[...,None]], axis=-1)
Reference=np.concatenate([X,F_sd[...,None]], axis=-1)


Particles = np.atleast_3d(Particles)
Reference = np.atleast_2d(Reference.T).T

variables = [2,0]

mean = np.einsum('inm,in->im', Particles, weights)

perturbations = Particles - mean[:,None,:]
std = np.sqrt(np.einsum('inm,in->im', perturbations**2, weights))

for i in [0,1]:
    if i == 0:
        col = imes_blue
    else:
        col = imes_green

    
    axs_overall[i+1,0].plot(time, mean[:,variables[i]], color=col, linestyle='-')
    
    axs_overall[i+1,0].fill_between(
            time, 
            mean[:,variables[i]] - 3*std[:,variables[i]], 
            mean[:,variables[i]] + 3*std[:,variables[i]], 
            facecolor=col, 
            edgecolor=None, 
            alpha=0.2
            )
    axs_overall[i+1,0].plot(time, Reference[:,variables[i]], color='k', linestyle=':')

    
    # set limits
    axs_overall[i+1,0].set_xlim(np.min(time), np.max(time))



axs_overall[1+1,0].set_ylabel(r"$s$ $\mathrm{in}$ $\mathrm{m}$")
# axs_overall[1+1,0].set_ylabel(r"$\dot{s}$ in $\mathrm{m/s}$")
axs_overall[0+1,0].set_ylabel(r"$F$ $\mathrm{in}$ $\mathrm{N}$")
axs_overall[0+1,0].set_ylim(-120,120)

axs_overall[1+1,0].set_xlabel(r"$\mathrm{Time}$ $\mathrm{in}$ $\mathrm{s}$")




################################################################################
################################################################################
################################# Vehicle ######################################
################################################################################
################################################################################

from src.Vehicle import basis_fcn, mu_y, GP_prior_f

################################################################################
# Loading

# Load the arrays from the file
loaded = np.load('VehicleSimulation_APF_saved.npz')

# Access individual arrays by their names
# Simulation
X = loaded['X']
Y = loaded['Y']
Mu_true = loaded['Mu_true']

# Particle filter
Sigma_X_veh = loaded['Sigma_X']
Sigma_mu_f = loaded['Sigma_mu_f']
Sigma_mu_r = loaded['Sigma_mu_r']
Sigma_alpha_f = loaded['Sigma_alpha_f']
Sigma_alpha_r = loaded['Sigma_alpha_r']
weights = loaded['weights']
Mean_f = loaded['Mean_f']
Col_Cov_f = loaded['Col_Cov_f']
Row_Scale_f = loaded['Row_Scale_f']
df_f = loaded['df_f']
Mean_r = loaded['Mean_r']
Col_Cov_r = loaded['Col_Cov_r']
Row_Scale_r = loaded['Row_Scale_r']
df_r = loaded['df_r']
fcn_var_f=loaded['fcn_var_f']
fcn_mean_f=loaded['fcn_mean_f']
fcn_var_r=loaded['fcn_var_r']
fcn_mean_r=loaded['fcn_mean_r']
alpha=loaded['alpha']
time=loaded['time']
steps=loaded['steps']


################################################################################
# Plotting


    
    
# plot time slices of the learned MTF for the front tire
alpha = jnp.linspace(-20/180*jnp.pi, 20/180*jnp.pi, 500)
mu_f_true = jax.vmap(functools.partial(mu_y))(alpha=alpha)
mu_r_true = jax.vmap(functools.partial(mu_y))(alpha=alpha)
basis_in = jax.vmap(basis_fcn)(alpha)

N_slices = 4
index2 = (np.array(range(N_slices))+1)/N_slices*(steps-1)
# index = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])*(steps-1)
index = np.array([0.05, 0.2, 0.4])*(steps-1)

# function values with GP prior
GP_prior = prior_mniw_2naturalPara_inv(
            GP_prior_f[0],
            GP_prior_f[1],
            GP_prior_f[2],
            GP_prior_f[3]
        )
_, col_scale_prior, row_scale_prior, _ = prior_mniw_Predictive(
    mean=GP_prior[0], 
    col_cov=GP_prior[1], 
    row_scale=GP_prior[2], 
    df=GP_prior[3], 
    basis=basis_in)
fcn_var_prior = np.diag(col_scale_prior-1) * row_scale_prior[0,0]
del col_scale_prior, row_scale_prior, GP_prior




# normalize variance to create transparency effect
fcn_alpha = np.maximum(np.minimum(1 - fcn_var_f/fcn_var_prior, 1), 0)

# generate plot
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        alpha, 
        Mean=fcn_mean_f[int(i)], 
        Std=np.sqrt(fcn_var_f[int(i)]),
        X_stats=Sigma_mu_f[:int(i)], 
        X_weights=weights[:int(i)],
        max_hist = 80)
    # ax_fcn_e[0][-1].set_xlabel(r"$\alpha$ $\mathrm{in}$ $\mathrm{rad}$")
    # ax_fcn_e[0][-1].set_ylabel(r"$\mu_f$ $\mathrm{in}$ $[-]$")
    ax_fcn_e[0][-1].set_ylim(-1.2,1.2)
    ax_fcn_e[0][-1].set_yticks([-1,0,1],['$-1$',r'$\mu_f$','$1$'])
    ax_fcn_e[0][-1].set_xticks([-0.2,0,0.2],['$-0.2$',r'$\alpha (\dot{\phi})$','$0.2$'])
    ax_fcn_e[1].text(0.14,35,r'$\# \mathrm{Data}$')
    # ax_fcn_e[1].set_ylabel(r"\# $\mathrm{Data}$")
    ax_fcn_e[0][-1].plot(alpha, mu_f_true, color='k', linestyle=':')
        
    apply_basic_formatting(fig_fcn_e, width=figsize_miniplots, height=1, font_size=fontsize_miniplots)
    # fig_fcn_e.tight_layout()
    fig_fcn_e.savefig(f"Vehicle_APF_muf_fcn_{np.round(time[int(i)],3)}.png",dpi=dpi_miniplots, bbox_inches='tight')




# plot weighted RMSE of GP over entire function space
fcn_var_f = fcn_var_f + 1e-4 # to avoid dividing by zero
v1 = np.sum(1/fcn_var_f, axis=-1)
wRMSE_f = np.sqrt(v1/(v1**2 - v1) * jnp.sum((fcn_mean_f - mu_f_true) ** 2 / fcn_var_f, axis=-1))

fcn_var_r = fcn_var_r + 1e-4 # to avoid dividing by zero
v1 = np.sum(1/fcn_var_r, axis=-1)
wRMSE_r = np.sqrt(v1/(v1**2 - v1) * jnp.sum((fcn_mean_r - mu_r_true) ** 2 / fcn_var_r, axis=-1))

# fig_RMSE, ax_RMSE = plt.subplots(1,1, layout='tight')
axs_overall[0,1].plot(
    time,
    wRMSE_f,
    color=imes_blue
)
axs_overall[0,1].plot(
    time,
    wRMSE_r,
    color=imes_orange
)
axs_overall[0,1].set_ylabel(r"$\mathrm{wRMSE}$ $\mathrm{in}$ $[-]$")
axs_overall[0,1].set_ylim(0)

for i in index:
    axs_overall[0,1].plot([time[int(i)], time[int(i)]], [0, wRMSE_f[int(i)]*1.5], color="black", linewidth=0.8)
    
    

# plot the state estimations
Particles=np.concatenate([Sigma_X_veh, Sigma_mu_f[...,None]], axis=-1)
Reference=np.concatenate([X,Mu_true[...,0,None]], axis=-1)

Particles = np.atleast_3d(Particles)
Reference = np.atleast_2d(Reference.T).T

variables = [2,0]

mean = np.einsum('inm,in->im', Particles, weights)

perturbations = Particles - mean[:,None,:]
std = np.sqrt(np.einsum('inm,in->im', perturbations**2, weights))

for i in [0,1]:
    if i == 0:
        col = imes_blue
    else:
        col = imes_green

    
    axs_overall[i+1,1].plot(time, mean[:,variables[i]], color=col, linestyle='-')
    axs_overall[i+1,1].fill_between(
            time, 
            mean[:,variables[i]] - 3*std[:,variables[i]], 
            mean[:,variables[i]] + 3*std[:,variables[i]], 
            facecolor=col, 
            edgecolor=None, 
            alpha=0.2
            )
    axs_overall[i+1,1].plot(time, Reference[:,variables[i]], color='k', linestyle=':')
    
    # set limits
    axs_overall[i+1,1].set_xlim(np.min(time), np.max(time))



axs_overall[1+1,1].set_ylabel(r"$\dot{\psi}$ $\mathrm{in}$ $\mathrm{rad}/\mathrm{s}$")
# axs_overall[1+1,1].set_ylabel(r"$v_y$ $\mathrm{in}$ $\mathrm{m}/\mathrm{s}$")
axs_overall[0+1,1].set_ylabel(r"$\mu_f$ $\mathrm{in}$ $[-]$")

axs_overall[1+1,1].set_xlabel(r"$\mathrm{Time}$ $\mathrm{in}$ $\mathrm{s}$")


################################################################################
################################################################################
################################# Battery ######################################
################################################################################
################################################################################

from src.Battery import basis_fcn, data, scale_C1, scale_R1
from src.Battery import offset_C1, offset_R1, GP_prior_C1R1

################################################################################
# Loading

# Load the arrays from the file
loaded = np.load('Battery_APF_saved.npz')


# Access individual arrays by their names
# Particle filter
Sigma_X_bat = loaded['Sigma_X']
Sigma_C1R1 = loaded['Sigma_C1R1']
Sigma_Y = loaded['Sigma_Y']
weights = loaded['weights']
Mean_C1R1 = loaded['Mean_C1R1']
Col_Cov_C1R1 = loaded['Col_Cov_C1R1']
Row_Scale_C1R1 = loaded['Row_Scale_C1R1']
df_C1R1 = loaded['df_C1R1']
fcn_var=loaded['fcn_var']
fcn_var_C1 = fcn_var[:,0,...]
fcn_var_R1 = fcn_var[:,1,...]
fcn_mean=loaded['fcn_mean']
fcn_mean_C1 = fcn_mean[:,0,...]
fcn_mean_R1 = fcn_mean[:,1,...]
time=loaded['time']
steps=loaded['steps']

# time conversion
time = time - time[0]
time = time.astype('float') * 1e-9 # ns to s
time = time / 3600 # s to h

################################################################################
# Plotting


# plot learned function
V = jnp.linspace(0, 2.2, 500)
basis_in = jax.vmap(basis_fcn)(V)

# N_slices = 4
# index2 = (np.array(range(N_slices))+1)/N_slices*(steps-1)
index = np.array([0.05, 0.15, 0.6])*(steps-1)

# generate slices plots
for i in index:
    fig_fcn_e, ax_fcn_e = plot_fcn_error_1D(
        V, 
        Mean=fcn_mean_C1[int(i)], 
        Std=np.sqrt(fcn_var_C1[int(i)]),
        X_stats=Sigma_X_bat[:int(i)], 
        X_weights=weights[:int(i)],
        max_hist = 3500)
    # ax_fcn_e[0][-1].set_xlabel(r"$U$ $\mathrm{in}$ $\mathrm{V}$")
    # ax_fcn_e[0][-1].set_ylabel(r"$C_1$ $\mathrm{in}$ $\mathrm{F}$")
    ax_fcn_e[0][-1].set_ylim(0,2200)
    ax_fcn_e[0][-1].set_yticks([0,1000,2000],['$0$',r'$C_1$','$2000$'])
    ax_fcn_e[0][-1].set_xticks([0,1,2],['$0$',r'$U$','$2$'])
    ax_fcn_e[1].text(0.1,1700,r'$\# \mathrm{Data}$')
    # ax_fcn_e[1].set_ylabel(r"\# $\mathrm{Data}$")

    # ax_fcn_e[0][1].set_xlabel(r"$U$ $\mathrm{in}$ $\mathrm{V}$")
    # ax_fcn_e[0][0].set_ylabel(r"$C_1$ $\mathrm{in}$ $\mathrm{F}$")
    # ax_fcn_e[0][1].set_ylabel(r"$R_1$ $\mathrm{in}$ $\mathrm{\Omega}$")

    apply_basic_formatting(fig_fcn_e, width=figsize_miniplots, height=1, font_size=fontsize_miniplots)
    # fig_fcn_e.tight_layout()
    fig_fcn_e.savefig(f"Battery_APF_C1R1_fcn_{int(i)}.png",dpi=dpi_miniplots, bbox_inches='tight')



# plot weighted RMSE of GP over entire function space
C1_true=fcn_mean_C1[-1,...]
R1_true=fcn_mean_R1[-1,...]

fcn_var_C1 = fcn_var_C1 + 1e-8 # to avoid dividing by zero
v1 = np.sum(1/fcn_var_C1, axis=-1)
# wRMSE_C1 = np.sqrt(v1/(v1**2 - v1) * jnp.sum((fcn_mean_C1 - C1_true[None,...]) ** 2 / fcn_var_C1, axis=-1))
RMSE_C1 = np.sqrt(jnp.sum((fcn_mean_C1 - C1_true[None,...]) ** 2/steps, axis=-1))


fcn_var_R1 = fcn_var_R1 + 1e-4 # to avoid dividing by zero
v1 = np.sum(1/fcn_var_R1, axis=-1)
# wRMSE_R1 = np.sqrt(v1/(v1**2 - v1) * jnp.sum((fcn_mean_R1 - R1_true[None,...]) ** 2 / fcn_var_R1, axis=-1))
RMSE_R1 = np.sqrt(jnp.sum((fcn_mean_R1 - R1_true[None,...]) ** 2/steps, axis=-1))


# fig_RMSE, ax_RMSE = plt.subplots(1,1, layout='tight')
axs_overall[0,2].plot(
    time,
    RMSE_C1,
    color=imes_blue
)
axs_overall[0,2].plot(
    time,
    RMSE_R1,
    color=imes_orange
)
axs_overall[0,2].set_ylabel(r"$\mathrm{RMSE}$ $\mathrm{in}$ $\mathrm{F}$ $\mathrm{and}$ $\mathrm{\Omega}$")
axs_overall[0,2].set_ylim(0)

for i in index:
    axs_overall[0,2].plot([time[int(i)], time[int(i)]], [0, RMSE_C1[int(i)]*1.5], color="black", linewidth=0.8)
    






# plot the state estimations
Particles=np.concatenate([Sigma_X_bat[...,None], (Sigma_C1R1 + np.array([offset_C1, offset_R1]))* np.array([scale_C1, scale_R1])], axis=-1)
# Reference=np.concatenate([X,F_sd[...,None]], axis=-1)

Particles = np.atleast_3d(Particles)
# Reference = np.atleast_2d(Reference.T).T

variables = [1,0]

mean = np.einsum('inm,in->im', Particles, weights)

perturbations = Particles - mean[:,None,:]
std = np.sqrt(np.einsum('inm,in->im', perturbations**2, weights))

for i in variables:
    if i == 0:
        col = imes_blue
    else:
        col = imes_green

    # axs_overall[i+1,0].plot(time, Reference[:,i], color=imes_blue)
    axs_overall[i+1,2].plot(time, mean[:,variables[i]], color=col, linestyle='-')
    
    axs_overall[i+1,2].fill_between(
            time, 
            mean[:,variables[i]] - 3*std[:,variables[i]], 
            mean[:,variables[i]] + 3*std[:,variables[i]], 
            facecolor=col, 
            edgecolor=None, 
            alpha=0.2
            )
    
    # set limits
    axs_overall[i+1,2].set_xlim(np.min(time), np.max(time))


axs_overall[1+1,2].set_ylabel(r"$U$ $\mathrm{in}$ $\mathrm{V}$")
axs_overall[0+1,2].set_ylabel(r"$C_1$ $\mathrm{in}$ $\mathrm{F}$")
axs_overall[0+1,2].set_ylim(850,2300)
# axs_overall[1+1,0].set_ylabel(r"$R_1$ in $\mathrm{\Omega}$")

axs_overall[1+1,2].set_xlabel(r"$\mathrm{Time}$ $\mathrm{in}$ $\mathrm{h}$")






#############################################################
# overall saving

apply_basic_formatting(fig_overall, width=17.78, height=1, font_size=fontsize_overall)
fig_overall.tight_layout()
fig_overall.savefig("grid_plot_online.svg", bbox_inches='tight')

