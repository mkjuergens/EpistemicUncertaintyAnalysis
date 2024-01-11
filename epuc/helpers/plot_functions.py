from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_gaussian_nig_prediction_intervals(
    results_dict: dict,
    x_train: np.ndarray,
    y_targets: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    eps_std: Optional[np.ndarray] = 3.0,
    figsize: tuple = (10, 21),
    list_subtitles: Optional[list] = None,
    plot_mean_params: bool = False
):
    """plot predcitions for first order loss minimisation (neg. log likelihood loss) as well as second order
    loss minimisation methods predicitng the parameters of the normal-inverse-gamma distribution.

    Parameters
    ----------
    results_dict : dict
        dictionary with training results for necessary parameters. See _simulation_gamma_nig
    x_train : np.ndarray
        array of training instances
    y_targets : np.ndarray
        array of training targets
    x_eval : np.ndarray
        array of instances where models are evaluated on
    y_eval : np.ndarray
        array of "true" targets,
    eps_std: np.ndarray or float
       standard deviation (aleatoric uncertainty) in the function generating process
    figsize : tuple, optional
        size of figure, by default (9, 21)
    list_subtitles : Optional[list], optional
        list of subtitles for subplots, by default None

    Returns
    -------
    fig, ax
    """
    n_cols = 3 if not plot_mean_params else 4
    fig, ax = plt.subplots(len(results_dict), n_cols, figsize=figsize)

    for i, ens_type in enumerate(results_dict.keys()):
        if not list_subtitles:
            subtitle = f"{ens_type}"
        else:
            subtitle = list_subtitles[i]
        fig.text(
            0.5,
            ax[i, 0].get_position().bounds[1] + ax[i, 0].get_position().height + 0.01,
            subtitle,
            ha="center",
            va="bottom",
            fontsize="large",
        )
        ax[i, 0].axvline(x_train.min(), linestyle="--", color="black", alpha=0.5)
        ax[i, 0].axvline(x_train.max(), linestyle="--", color="black", alpha=0.5)
        ax[i, 1].axvline(x_train.min(), linestyle="--", color="black", alpha=0.5)
        ax[i, 1].axvline(x_train.max(), linestyle="--", color="black", alpha=0.5)
        ax[i, 0].set_title("$\mu$")
        ax[i, 1].set_title("$\sigma^2$")
        ax[i, 2].set_title("predicted parameters")
        if plot_mean_params:
            ax[i, 3].set_title("mean parameters")

        ax[i, 0].plot()
        ax[i, 0].plot(x_eval, y_eval, label="ground truth", color="red")
        # plot horizontal line at eps_std **2
        ax[i, 1].axhline(
            eps_std**2, linestyle="--", color="red", label="grounds truth $sigma^2$"
        )
        ax[i,0].set_xlabel("x")
        ax[i,1].set_xlabel("x")
        ax[i,2].set_xlabel("x")

        if ens_type == "Normal":
            # plot predictions for mu ------------------------------------
            ax[i, 0].plot(
                x_eval,
                results_dict[ens_type]["mean_mus"],
                label="mean prediction",
                color="blue",
            )
            ax[i, 0].plot(
                x_eval, results_dict[ens_type]["pred_mus"], alpha=0.1, color="black"
            )
            ax[i, 0].fill_between(
                x_eval,
                results_dict[ens_type]["lower_mu"],
                results_dict[ens_type]["upper_mu"],
                alpha=0.5,
                label="95% CI",
                color="gray",
            )
            ax[i, 0].scatter(
                x_train,
                y_targets,
                label="training data",
                marker="o",
                s=20,
                color="black",
                alpha=0.1,
            )
            ax[i, 0].legend()
            # ------------------------------------
            # plot predictions for sigma
            ax[i, 1].plot(
                x_eval,
                results_dict[ens_type]["pred_sigmas2"],
                alpha=0.1,
                color="black",
            )
            ax[i, 1].plot(
                x_eval,
                results_dict[ens_type]["mean_sigma2"],
                label="mean prediction",
                color="blue",
            )
            ax[i, 1].fill_between(
                x_eval,
                results_dict[ens_type]["lower_sigma"],
                results_dict[ens_type]["upper_sigma"],
                alpha=0.5,
                label="95% CI",
                color="gray",
            )
            ax[i, 1].legend()
            # ------------------------------------
            # plot predicted parameters for the normal distribution
            ax[i, 2].plot(
                x_eval,
                results_dict[ens_type]["mean_mus"],
                label=r"$\mu$",
                color="blue",
            )
            ax[i, 2].plot(
                x_eval,
                results_dict[ens_type]["mean_sigma"],
                label=r"$\sigma$",
            )
            ax[i, 2].legend()
            # ------------------------------------
            # plotmean output parameters if desired
            if plot_mean_params:
                ax[i, 3].plot(results_dict[ens_type]["param_0"], label=r"$\mu$")
                ax[i, 3].plot(results_dict[ens_type]["param_1"], label=r"$\sigma$")
                ax[i, 3].set_xlabel("epoch")
                ax[i, 3].legend()



        else:
            # plot predictions for mu ------------------------------------
            ax[i, 0].plot(
                x_eval,
                results_dict[ens_type]["mean_pred_mu"],
                label="mean prediction",
                color="blue",
            )
            ax[i, 0].fill_between(
                x_eval,
                results_dict[ens_type]["lower_mu"],
                results_dict[ens_type]["upper_mu"],
                alpha=0.5,
                label="95% CI",
                color="blue",
            )
            ax[i, 0].scatter(
                x_train,
                y_targets,
                marker="x",
                s=10,
                label="targets",
                color="black",
                alpha=0.4,
            )
            ax[i, 0].legend()
            # ------------------------------------
            # plot predictions for sigma ------------------------------------
            ax[i, 1].plot(
                x_eval,
                results_dict[ens_type]["mean_pred_sigma2"],
                label="mean prediction",
                color="blue",
            )
            ax[i, 1].fill_between(
                x_eval,
                results_dict[ens_type]["lower_sigma"],
                results_dict[ens_type]["upper_sigma"],
                alpha=0.5,
                label="95% CI",
                color="blue",
            )
            ax[i, 1].legend()
            # ------------------------------------
            # plot predicted parameters for the normal-inverse-gamma distribution
            ax[i, 2].plot(
                x_eval,
                np.mean(results_dict[ens_type]["pred_gammas"], axis=1),
                label=r"$\gamma$",
                color="blue",
            )
            ax[i, 2].plot(
                x_eval,
                np.mean(results_dict[ens_type]["pred_nus"], axis=1),
                label=r"$\nu$",
                color="red",
            )
            ax[i, 2].plot(
                x_eval,
                np.mean(results_dict[ens_type]["pred_alphas"], axis=1),
                label=r"$\alpha$",
                color="green",
            )
            ax[i, 2].plot(
                x_eval,
                np.mean(results_dict[ens_type]["pred_betas"], axis=1),
                label=r"$\beta$",
                color="orange",
            )
            ax[i, 2].legend()
            # ------------------------------------
            # plot mean output parameters if desired
            if plot_mean_params:
                ax[i, 3].plot(results_dict[ens_type]["param_0"], label=r"$\gamma$")
                ax[i, 3].plot(results_dict[ens_type]["param_1"], label=r"$\nu$")
                ax[i, 3].plot(results_dict[ens_type]["param_2"], label=r"$\alpha$")
                ax[i, 3].plot(results_dict[ens_type]["param_3"], label=r"$\beta$")
                ax[i, 3].set_xlabel("epoch")
                ax[i, 3].legend()


    fig.subplots_adjust(hspace=0.5)
    return fig, ax


def plot_bernoulli_beta_prediction_intervals(
    results_dict: dict,
    x_train: np.ndarray,
    y_targets: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    figsize: tuple = (9, 21),
    plot_mean_params: bool = False
):
    
    n_cols = 2 if not plot_mean_params else 3
    fig, ax = plt.subplots(len(results_dict), n_cols, figsize=figsize)
    for i, ens_type in enumerate(results_dict.keys()):
        fig.text(
            0.5,
            ax[i, 0].get_position().bounds[1] + ax[i, 0].get_position().height + 0.01,
            f"{ens_type}",
            ha="center",
            va="bottom",
            fontsize="large",
        )
        ax[i, 0].axvline(x_train.min(), linestyle="--", color="black")
        ax[i, 0].axvline(x_train.max(), linestyle="--", color="black")
        ax[i, 0].set_title(r"$\theta$")
        ax[i, 1].set_title("predicted parameters")
        if plot_mean_params:
            ax[i, 2].set_title("mean parameters")
        ax[i, 0].set_xlabel("x")
        ax[i, 1].set_xlabel("x")
        ax[i, 0].plot(x_eval, y_eval, label=r"ground truth $\theta$", color="red")
        # plot training data
        ax[i, 0].scatter(
            x_train,
            y_targets,
            label="training data",
            marker="o",
            s=20,
            color="black",
            alpha=0.1,
        )
        if ens_type == "Bernoulli":
            # plot predictions for mu ------------------------------------
            ax[i, 0].plot(
                x_eval,
                results_dict[ens_type]["mean_probs"],
                label="mean prediction",
                color="blue",
            )
            ax[i, 0].plot(
                x_eval,
                results_dict[ens_type]["pred_probs"][:, :, 0],
                alpha=0.1,
                color="black",
            )
            ax[i, 0].fill_between(
                x_eval,
                results_dict[ens_type]["lower_p"],
                results_dict[ens_type]["upper_p"],
                alpha=0.5,
                label="95% CI",
                color="gray",
            )
            ax[i, 0].legend()
            # ------------------------------------
            # plot predictions for theta
            ax[i, 1].plot(
            x_eval,
            results_dict[ens_type]["mean_probs"],
            label=r"$\theta$",
            color="blue",
                )
            
            ax[i, 1].legend()
            # ------------------------------------
            # plot mean output parameters if desired
            if plot_mean_params:
                ax[i, 2].plot(results_dict[ens_type]["param_0"], label=r"$\theta$")
                ax[i, 2].set_xlabel("epoch")
                ax[i, 2].legend()

        else:
            # plot predictions for mu ------------------------------------
            ax[i, 0].plot(
                x_eval,
                results_dict[ens_type]["mean_pred_p"],
                label="mean prediction",
                color="blue",
            )
            ax[i, 0].fill_between(
                x_eval,
                results_dict[ens_type]["lower_p"],
                results_dict[ens_type]["upper_p"],
                alpha=0.5,
                label="95% CI",
                color="blue",
            )

            ax[i, 0].legend()
            # ------------------------------------
            # plot predicted parameters for the beta distribution
            ax[i, 1].plot(
                x_eval,
                np.mean(results_dict[ens_type]["pred_alphas"], axis=1),
                label=r"$\alpha$",
                color="blue",
            )
            ax[i, 1].plot(
                x_eval,
                np.mean(results_dict[ens_type]["pred_betas"], axis=1),
                label=r"$\beta$",
                color="red",
            )
            ax[i, 1].legend()
            # ------------------------------------
            # plot mean output parameters if desired
            if plot_mean_params:
                ax[i, 2].plot(results_dict[ens_type]["param_0"], label=r"$\alpha$")
                ax[i, 2].plot(results_dict[ens_type]["param_1"], label=r"$\beta$")
                ax[i, 2].set_xlabel("epoch")
                ax[i, 2].legend()

    fig.subplots_adjust(hspace=0.5)
    return fig, ax
