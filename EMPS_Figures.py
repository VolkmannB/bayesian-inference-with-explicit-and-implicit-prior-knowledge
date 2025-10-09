import numpy as np
import jax
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt

from src.EMPS import steps
from src.Publication_Plotting import plot_fcn_error_1D, imes_blue
from src.Publication_Plotting import apply_basic_formatting, plot_Data
from src.BayesianInferrence import prior_mniw_Predictive
from src.BayesianInferrence import prior_mniw_2naturalPara_inv


################################################################################
# Model

N_slices = 2


data = scipy.io.loadmat("plots/EMPS.mat")


offline_Sigma_X = data["offline_Sigma_X"]
offline_Sigma_F = data["offline_Sigma_F"]
offline_Sigma_Y = data["offline_Sigma_Y"]
offline_weights = data["offline_weights"]
offline_T0 = data["offline_T0"]
offline_T1 = data["offline_T1"]
offline_T2 = data["offline_T2"]
offline_T3 = data["offline_T3"].flatten()

online_Sigma_X = data["online_Sigma_X"]
online_Sigma_F = data["online_Sigma_F"]
online_Sigma_Y = data["online_Sigma_Y"]
online_weights = data["online_weights"]
online_T0 = data["online_T0"]
online_T1 = data["online_T1"]
online_T2 = data["online_T2"]
online_T3 = data["online_T3"].flatten()

GP_prior_stats = [
    data["prior_T0"],
    data["prior_T1"],
    data["prior_T2"],
    data["prior_T3"].flatten(),
]

dq_plot = data["dq_plot"].flatten()
basis_plot = data["basis_plot"]
time = data["time"].flatten()

Y = data["Y"].flatten()
X = data["X"]

del data


# Convert sufficient statistics to standard parameters
(offline_GP_Mean, offline_GP_Col_Cov, offline_GP_Row_Scale, offline_GP_df) = (
    jax.vmap(prior_mniw_2naturalPara_inv)(
        GP_prior_stats[0]
        + np.cumsum(offline_T0, axis=0)
        / np.arange(1, offline_Sigma_X.shape[1] + 1)[:, None, None],
        GP_prior_stats[1]
        + np.cumsum(offline_T1, axis=0)
        / np.arange(1, offline_Sigma_X.shape[1] + 1)[:, None, None],
        GP_prior_stats[2]
        + np.cumsum(offline_T2, axis=0)
        / np.arange(1, offline_Sigma_X.shape[1] + 1)[:, None, None],
        GP_prior_stats[3]
        + np.cumsum(offline_T3, axis=0)
        / np.arange(1, offline_Sigma_X.shape[1] + 1),
    )
)
del offline_T0, offline_T1, offline_T2, offline_T3

# Convert sufficient statistics to standard parameters
(online_GP_Mean, online_GP_Col_Cov, online_GP_Row_Scale, online_GP_df) = (
    jax.vmap(prior_mniw_2naturalPara_inv)(
        GP_prior_stats[0] + online_T0,
        GP_prior_stats[1] + online_T1,
        GP_prior_stats[2] + online_T2,
        GP_prior_stats[3] + online_T3,
    )
)
del online_T0, online_T1, online_T2, online_T3


################################################################################
# Plotting Offline

# plot the state estimations
fig_X, axes_X = plt.subplots(3, 1, layout="tight", sharex="col", dpi=150)
plot_Data(
    Particles=np.concatenate([offline_Sigma_X, offline_Sigma_F], axis=-1),
    weights=offline_weights,
    Reference=np.concatenate([X, np.ones((Y.shape[0], 1)) * np.nan], axis=-1),
    time=time,
    axes=axes_X,
)
axes_X[0].set_title("Offline")
axes_X[0].set_ylabel(r"$q$ in m")
axes_X[1].set_ylabel(r"$\dot{q}$ in m/s")
axes_X[2].set_ylabel(r"$F$ in N")
axes_X[2].set_xlabel(r"Time in s")
axes_X[2].set_ylim(-100, 100)
apply_basic_formatting(fig_X, width=8, height=16, font_size=8)
fig_X.savefig("plots/EMPS_PGAS_X.pdf", bbox_inches="tight")

N_PGAS_iter = offline_Sigma_X.shape[1]
index = (np.array(range(N_slices)) + 1) / N_slices * (N_PGAS_iter - 1)

# function value from GP
fcn_mean_offline = np.zeros((N_PGAS_iter, dq_plot.shape[0]))
fcn_var_offline = np.zeros((N_PGAS_iter, dq_plot.shape[0]))
for i in tqdm(range(0, N_PGAS_iter), desc="Calculating fcn value and var"):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=offline_GP_Mean[i],
        col_cov=offline_GP_Col_Cov[i],
        row_scale=offline_GP_Row_Scale[i],
        df=offline_GP_df[i],
        basis=basis_plot,
    )
    fcn_var_offline[i] = np.diag(col_scale - 1) * row_scale[0, 0]
    fcn_mean_offline[i] = mean

# generate plot
c = 0
for i in index:
    N_tasks = 1
    fig_fcn_e = plt.figure(dpi=150)
    gs = fig_fcn_e.add_gridspec(
        N_tasks + 1,
        1,
        height_ratios=(1, *(5 * np.ones((N_tasks,)))),
        hspace=0.05,
        wspace=0.05,
    )
    ax = [fig_fcn_e.add_subplot(gs[i + 1, 0]) for i in range(0, N_tasks)]
    ax_histx = fig_fcn_e.add_subplot(gs[0, 0], sharex=ax[-1])

    plot_fcn_error_1D(
        dq_plot,
        Mean=fcn_mean_offline[int(i)],
        Std=np.sqrt(fcn_var_offline[int(i)]),
        X_stats=offline_Sigma_X[:, : int(i), 1],
        X_weights=offline_weights[:, : int(i)],
        ax=ax,
        ax_histx=ax_histx,
    )

    ax[-1].set_xlabel(r"$\dot{q}$ in m/s")
    ax[-1].set_ylabel(r"$F$ in N")
    ax[-1].set_ylim(-65, 65)
    fig_fcn_e.suptitle(f"Iteration {int(i + 1)}", fontsize=8)

    apply_basic_formatting(fig_fcn_e, width=8, height=8, font_size=8)
    fig_fcn_e.savefig(
        f"plots/EMPS_PGAS_F_fcn_{int(c)}.pdf", bbox_inches="tight"
    )
    c += 1


# plot RMSE between observations and PGAS iterations
RMSE = np.zeros((N_PGAS_iter,))
for i in range(0, N_PGAS_iter):
    RMSE[i] = np.sqrt(np.mean((Y - offline_Sigma_Y[:, i]) ** 2))
fig_RMSE, ax_RMSE = plt.subplots(1, 1, layout="tight")
ax_RMSE.plot(range(0, N_PGAS_iter), RMSE, color=imes_blue)
ax_RMSE.set_xlabel(r"Iterations")
ax_RMSE.set_ylabel(r"RMSE in m")
apply_basic_formatting(fig_RMSE, width=8, height=8, font_size=8)
fig_RMSE.savefig("plots/EMPS_PGAS_RMSE.pdf", bbox_inches="tight")


################################################################################
# Plotting Online

# plot the state estimations
fig_X, axes_X = plt.subplots(3, 1, layout="tight", sharex="col", dpi=150)
plot_Data(
    Particles=np.concatenate([online_Sigma_X, online_Sigma_F], axis=-1),
    weights=online_weights,
    Reference=np.concatenate([X, np.ones((Y.shape[0], 1)) * np.nan], axis=-1),
    time=time,
    axes=axes_X,
)
axes_X[0].set_title("Online")
axes_X[0].set_ylabel(r"$q$ in m")
axes_X[1].set_ylabel(r"$\dot{q}$ in m/s")
axes_X[2].set_ylabel(r"$F$ in N")
axes_X[2].set_xlabel(r"Time in s")
axes_X[2].set_ylim(-100, 100)
apply_basic_formatting(fig_X, width=8, height=16, font_size=8)
fig_X.savefig("plots/EMPS_APF_X.pdf", bbox_inches="tight")

index = (np.array(range(N_slices)) + 1) / N_slices * (steps - 1)

# function value from GP
fcn_mean_online = np.zeros((steps, dq_plot.shape[0]))
fcn_var_online = np.zeros((steps, dq_plot.shape[0]))
for i in tqdm(range(0, steps), desc="Calculating fcn value and var"):
    mean, col_scale, row_scale, _ = prior_mniw_Predictive(
        mean=online_GP_Mean[i],
        col_cov=online_GP_Col_Cov[i],
        row_scale=online_GP_Row_Scale[i],
        df=online_GP_df[i],
        basis=basis_plot,
    )
    fcn_var_online[i] = np.diag(col_scale - 1) * row_scale[0, 0]
    fcn_mean_online[i] = mean

# generate plot
c = 0
for i in index:
    N_tasks = 1
    fig_fcn_e = plt.figure(dpi=150)
    gs = fig_fcn_e.add_gridspec(
        N_tasks + 1,
        1,
        height_ratios=(1, *(5 * np.ones((N_tasks,)))),
        hspace=0.05,
        wspace=0.05,
    )
    ax = [fig_fcn_e.add_subplot(gs[i + 1, 0]) for i in range(0, N_tasks)]
    ax_histx = fig_fcn_e.add_subplot(gs[0, 0], sharex=ax[-1])

    plot_fcn_error_1D(
        dq_plot,
        Mean=fcn_mean_online[int(i)],
        Std=np.sqrt(fcn_var_online[int(i)]),
        X_stats=online_Sigma_X[: int(i), :, 1],
        X_weights=online_weights[: int(i)],
        ax=ax,
        ax_histx=ax_histx,
    )

    ax[-1].set_xlabel(r"$\dot{q}$ in m/s")
    ax[-1].set_ylabel(r"$F$ in N")
    ax[-1].set_ylim(-65, 65)
    fig_fcn_e.suptitle(f"Time {np.round(time[int(i)], 2)} s", fontsize=8)

    apply_basic_formatting(fig_fcn_e, width=8, height=8, font_size=8)
    fig_fcn_e.savefig(f"plots/EMPS_APF_F_fcn_{int(c)}.pdf", bbox_inches="tight")
    c += 1


# plot weighted RMSE of GP over entire function space
fcn_var = fcn_var_online + fcn_var_offline[-1]
w = 1 / fcn_var
v1 = np.sum(w, axis=-1)
v2 = np.sum(w**2, axis=-1)
wRMSE = np.sqrt(
    1
    / (v1 - (v2 / v1**2))
    * np.sum((fcn_mean_online - fcn_mean_offline[-1]) ** 2 * w, axis=-1)
)

fig_RMSE, ax_RMSE = plt.subplots(1, 1, layout="tight")
ax_RMSE.plot(time, wRMSE, color=imes_blue)
ax_RMSE.set_ylabel(r"wRMSE")
ax_RMSE.set_xlabel(r"Time in $\mathrm{s}$")
ax_RMSE.set_ylim(0)

for i in index:
    ax_RMSE.plot(
        [time[int(i)], time[int(i)]],
        [0, wRMSE[int(i)] * 1.5],
        color="black",
        linewidth=0.8,
    )


apply_basic_formatting(fig_RMSE, width=8, height=8, font_size=8)
fig_RMSE.savefig("plots/EMPS_APF_F_wRMSE.pdf", bbox_inches="tight")


plt.show()
