from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_gaussian_nig_prediction_intervals(
    results_dict: dict,
    x_train: np.ndarray,
    y_targets: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    figsize: tuple = (9, 21),
    list_subtitles: Optional[list] = None,
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
        array of "true" targets
    figsize : tuple, optional
        size of figure, by default (9, 21)
    list_subtitles : Optional[list], optional
        list of subtitles for subplots, by default None

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(len(results_dict), 2, figsize=figsize)

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
        ax[i, 0].set_title("$\mu$")
        ax[i, 1].set_title("$\sigma$")
        ax[i, 0].plot()
        ax[i, 0].plot(x_eval, y_eval, label="ground truth", color="red")

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

    fig.subplots_adjust(hspace=0.5)
    return fig, ax
