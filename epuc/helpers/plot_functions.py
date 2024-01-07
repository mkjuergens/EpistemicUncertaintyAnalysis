from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_gaussian_nig_prediction_intervals(results_dict: dict, x_train: np.ndarray, y_targets: np.ndarray,
                                           x_eval: np.darray, y_eval: np.ndarray, title: str, figsize: tuple, 
                                           list_subtitles: Optional[list] = None):
    
    fig, ax = plt.subplots(len(results_dict), 2, figsize=figsize)

    for i, ens_type in enumerate(results_dict.keys()):
        fig.text(
        0.5,
        ax[i, 0].get_position().bounds[1] + ax[i, 0].get_position().height + 0.01,
        f"{ens_type}",
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
                results_dict[ens_type]["mean_mus"].detach().numpy(),
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
                results_dict[ens_type]["pred_sigmas"],
                alpha=0.1,
                color="black",
            )
            ax[i, 1].plot(
                x_eval,
                results_dict[ens_type]["mean_sigmas"].detach().numpy(),
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
                results_dict[ens_type]["mean_pred_mu"].detach().numpy(),
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
                results_dict[ens_type]["mean_pred_sigma2"].detach(),
                label="mean prediction", color="blue",
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