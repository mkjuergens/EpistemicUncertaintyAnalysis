import json
import os
import argparse
from typing import Optional
import datetime
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from epuc.datasets import create_evaluation_data

# from epuc.helpers.ensemble import Ensemble, GaussianEnsemble, NIGEnsemble, BetaEnsemble
from epuc.configs import data_config, create_train_config
from epuc.uncertainty import (
    get_upper_lower_bounds_normal,
    get_upper_lower_bounds_empirical,
    get_upper_lower_bounds_inv_gamma,
    get_upper_lower_bounds_beta,
)

from epuc.helpers.plot_functions import (
    plot_gaussian_nig_prediction_intervals,
    plot_bernoulli_beta_prediction_intervals,
)

plt.style.use("seaborn-v0_8")


def _main_simulation(
    config_dir,
    ens_type: Optional[str] = None,
    type: str = "regression",
    data_type: str = "polynomial",
    exp_name: Optional[list] = None,
    save_dir: str = "results",
    return_mean_params: bool = False,
    return_std_params: bool = False,
    plot_results: bool = True
):
    """function for doing the primary-secondary distribution analysis, saving the results in a
    dictionary and plotting it.

    Parameters
    ----------
    config_dir : str
        directory where
    ens_type: str, Optional
        type of ensemble that is to be trained. If None, every ensemble in the config
    type : str, optional
        type of the experiment, by default "regression"
    data_type : str, optional
        type of data, by default "polynomial"
    dataset : _type_, optional
        _description_, by default PolynomialDataset
    exp_name : Optional[str], optional
        _description_, by default None
    return_mean_params: bool
        whether to return the mean outputs per training epoch
    """

    if not exp_name:
        exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # load json file located in the config_dir directory into a dictionary
    with open(config_dir) as json_file:
        temp_dict = json.load(json_file)

    save_path = f"{save_dir}/" + type + f"/{exp_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "/params.json", "w") as outfile:
        json.dump(temp_dict, outfile)

    dataset, x_eval, y_eval, x_train, y_targets = create_evaluation_data(
        data_config=data_config,
        problem_type=type,
        data_type=data_type,
        n_eval_points=1000,
    )

    train_config = create_train_config(type=type, **temp_dict)

    results_per_ens_dict = {}

    if ens_type:
        keys = list(ens_type)
    else:
        keys = train_config.keys()
    for ens_type in keys:
        # crate dictionary for each ensemble type
        results = {}
        #results = {}

        ensemble = train_config[ens_type]["ensemble"](
            model_config=train_config[ens_type]["model_config"],
            ensemble_size=train_config[ens_type]["ensemble_size"],
        )
        ensemble.train(
            dataset=dataset,
            data_params=data_config[type][data_type],
            train_params=train_config[ens_type],
            return_mean_params=return_mean_params,
            return_std_params=return_std_params,
            x_eval=x_eval,
        )
        if return_mean_params:
            for key in ensemble.dict_mean_params.keys():
                results[key] = ensemble.dict_mean_params[key]
        if return_std_params:
            for key in ensemble.dict_std_params.keys():
                results[f'{key}_std'] = ensemble.dict_std_params[key]

        preds = ensemble.predict(x_eval.view(-1, 1)).detach().numpy()

        # get mean and standard deviation of both mu and sigma predictions TODO: will probably need to be customized!
        mean_params = ensemble.predict_mean(x_eval.view(-1, 1)).detach().numpy()

        # results["mean_params"] = mean_params
        # results["std_params"] = std_params

        if type == "classification":
            if ens_type == "Bernoulli":

                results["mean_probs"] = mean_params
                results["pred_probs"] = preds

                # confidence bounds
                lower_p, upper_p = get_upper_lower_bounds_empirical(
                    p=0.975, y=preds[:, :, 0]
                )

                results["lower_p"] = lower_p
                results["upper_p"] = upper_p

            else:
                results["pred_alphas"] = preds[:, :, 0]
                results["pred_betas"] = preds[:, :, 1]

                results["mean_pred_p"] = (
                    ensemble.predict_mean_p(x_eval.view(-1, 1)).detach().numpy()
                )

                # confidence bounds
                lower_p, upper_p = get_upper_lower_bounds_beta(
                    p=0.975, alpha=mean_params[:, 0], beta=mean_params[:, 1]
                )
                results["lower_p"] = lower_p
                results["upper_p"] = upper_p

        elif type == "regression":
            if ens_type == "Normal":
                std_params = ensemble.predict_std(x_eval.view(-1, 1)).detach().numpy()

                results["mean_mus"] = mean_params[:, 0]
                # take the square of the standard deviation to get the variance
                results["mean_sigma2"] = np.mean(
                    (preds[:, :, 1] ** 2), axis=1
                )
                results["mean_sigma"] = np.mean(
                    preds[:, :, 1], axis=1
                )
                # results["mean_sigmas"] = mean_params[:, 1]
                results["pred_mus"] = preds[:, :, 0]
                results[ens_type]["pred_sigmas2"] = preds[:, :, 1] ** 2

                # confidence bounds
                lower_mu, upper_mu = get_upper_lower_bounds_normal(
                    p=0.975,
                    mu=mean_params[:, 0],
                    sigma=std_params[:, 0],
                )
                lower_sigma2, upper_sigma2 = get_upper_lower_bounds_empirical(
                    p=0.975,
                    y=preds[:, :, 1] ** 2,
                )

                results["lower_mu"] = lower_mu
                results["upper_mu"] = upper_mu
                results["lower_sigma"] = lower_sigma2
                results["upper_sigma"] = upper_sigma2

            else:
                (
                    mean_mu,
                    var_mu,
                    mean_sigma2,
                    var_sigma2,
                ) = ensemble.predict_normal_params(x_eval.view(-1, 1))

                results["pred_gammas"] = preds[:, :, 0]
                results["pred_nus"] = preds[:, :, 1]
                results["pred_alphas"] = preds[:, :, 2]
                results["pred_betas"] = preds[:, :, 3]

                results[
                    "mean_pred_mu"
                ] = mean_mu.detach().numpy()
                results[
                    "mean_pred_sigma2"
                ] = mean_sigma2.detach().numpy()

                # confidence bounds
                lower_mu, upper_mu = get_upper_lower_bounds_normal(
                    p=0.975,
                    mu=mean_mu.detach().numpy(),
                    sigma=np.sqrt(var_mu.detach().numpy()),
                )
                # take alpha and beta as parameters for inverse gamma distribution to get bounds for sigma2
                lower_sigma2, upper_sigma2 = get_upper_lower_bounds_inv_gamma(
                    p=0.975,
                    alpha=mean_params[:, 2],
                    beta=mean_params[:, 3],
                )

                results["lower_mu"] = lower_mu
                results["upper_mu"] = upper_mu
                results["lower_sigma"] = lower_sigma2
                results["upper_sigma"] = upper_sigma2

                # save results dict
        results_per_ens_dict[ens_type] = results
        with open(save_path + f"/results_{ens_type}.pkl", "wb") as f:
                    pickle.dump(results, f)

    # save end results in a pickle file
    with open(save_path + "/results_per_ens_dict.pkl", "wb") as f:
        pickle.dump(results_per_ens_dict, f)

    if plot_results:
        # plot results
        if type == "classification":
            if return_mean_params:
                figsize = (13, 21)
            else:
                figsize = (10, 21)
            fig, ax = plot_bernoulli_beta_prediction_intervals(
                results_dict=results_per_ens_dict,
                x_train=x_train,
                x_eval=x_eval,
                y_targets=y_targets,
                y_eval=y_eval,
                figsize=figsize,
                plot_mean_params=return_mean_params,
            )
        elif type == "regression":
            if return_mean_params:
                figsize = (15, 21)
            else:
                figsize = (11, 21)
            eps_std = data_config["regression"][data_type]["eps_std"]
            fig, ax = plot_gaussian_nig_prediction_intervals(
                results_dict=results_per_ens_dict,
                x_train=x_train,
                y_targets=y_targets,
                x_eval=x_eval,
                y_eval=y_eval,
                eps_std=eps_std,
                figsize=figsize,
                plot_mean_params=return_mean_params,
            )

        # save plot in same folder
        plt.savefig(save_path + "/fig_confidence_bounds.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", dest="config_dir", type=str, required=True)
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="results")
    parser.add_argument("--type", dest="type", type=str, default="regression")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--data_type", dest="data_type", type=str, required=True)
    parser.add_argument(
        "--return_mean_params", dest="return_mean_params", type=bool, default=True
    )
    parser.add_argument(
        "--return_std_params", dest="return_std_params", type=bool, default=True
    )
    parser.add_argument("--plot_results", dest="plot_results", type=bool, default=True)
    parser.add_argument("--ens_type", nargs="*", dest="ens_type", default=None)

    args = parser.parse_args()
    _main_simulation(
        config_dir=args.config_dir,
        ens_type=args.ens_type,
        type=args.type,
        data_type=args.data_type,
        save_dir=args.save_dir,
        exp_name=args.exp_name,
        return_mean_params=args.return_mean_params,
        return_std_params=args.return_std_params,
    )
