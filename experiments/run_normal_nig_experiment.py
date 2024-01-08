import json
import os
import argparse
from typing import Optional
import datetime
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from epuc.datasets import (
    SineRegressionDataset,
    PolynomialDataset,
    polynomial_fct,
    sine_fct_prediction,
)
from epuc.losses import inner_loss_der
from epuc.helpers.ensemble import Ensemble, GaussianEnsemble, NIGEnsemble
from epuc.configs import model_config, data_config, create_train_config
from epuc.uncertainty import (
    get_upper_lower_bounds_normal,
    get_upper_lower_bounds_empirical,
    get_upper_lower_bounds_inv_gamma,
)

plt.style.use("seaborn-v0_8")


def _simulation_gamma_nig(
    config_dir,
    type: str = "regression",
    dataset=PolynomialDataset,
    exp_name: Optional[str] = None,
):
    if not exp_name:
        exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    dataset_eval = PolynomialDataset(
        **data_config["PolynomialRegression"]
    )  # TODO: generalize

    x_eval = torch.from_numpy(np.linspace(-6, 6, 100)).float()
    y_eval = polynomial_fct(x_eval, degree=3)
    x_inst = dataset_eval.x_inst
    y_targets = dataset_eval.y_targets

    if not os.path.exists("results/" + exp_name):
        os.makedirs("results/" + exp_name)

    # load json file located in the config_dir directory into a dictionary
    with open(config_dir) as json_file:
        temp_dict = json.load(json_file)

    with open("results/" + exp_name + "/sample.json", "w") as outfile:
        json.dump(temp_dict, outfile)

    train_config = create_train_config(type=type, **temp_dict)

    results_per_ens_dict = {}
    for ens_type in train_config.keys():
        results_per_ens_dict[ens_type] = {}
        if ens_type == "Normal":
            ensemble = GaussianEnsemble(
                model_config=model_config["Normal"],
                ensemble_size=temp_dict["ensemble_size"],
            )

        else:
            ensemble = NIGEnsemble(
                model_config=model_config["NormalInverseGamma"],
                ensemble_size=temp_dict["ensemble_size"],
            )

        ensemble.train(
            dataset=PolynomialDataset,
            data_params=data_config["PolynomialRegression"],
            train_params=train_config[ens_type],
        )

        preds = ensemble.predict(x_eval.view(-1, 1)).detach().numpy()

        # get mean and standard deviation of both mu and sigma predictions TODO: will probably need to be customized!
        mean_params = ensemble.predict_mean(x_eval.view(-1, 1))

        # results_per_ens_dict[ens_type]["mean_params"] = mean_params
        # results_per_ens_dict[ens_type]["std_params"] = std_params

        if ens_type == "Normal":
            std_params = ensemble.predict_std(x_eval.view(-1, 1))

            results_per_ens_dict[ens_type]["mean_mus"] = mean_params[:, 0]
            results_per_ens_dict[ens_type]["mean_sigmas"] = mean_params[:, 1]
            results_per_ens_dict[ens_type]["pred_mus"] = preds[:, :, 0]
            results_per_ens_dict[ens_type]["pred_sigmas"] = preds[:, :, 1]

            # confidence bounds
            lower_mu, upper_mu = get_upper_lower_bounds_normal(
                p=0.975,
                mu=mean_params[:, 0].detach().numpy(),
                sigma=std_params[:, 0].detach().numpy(),
            )
            lower_sigma2, upper_sigma2 = get_upper_lower_bounds_empirical(
                p=0.975,
                y=preds[:, :, 1] ** 2,
            )

            results_per_ens_dict[ens_type]["lower_mu"] = lower_mu
            results_per_ens_dict[ens_type]["upper_mu"] = upper_mu
            results_per_ens_dict[ens_type]["lower_sigma"] = lower_sigma2
            results_per_ens_dict[ens_type]["upper_sigma"] = upper_sigma2

        else:
            mean_mu, var_mu, mean_sigma2, var_sigma2 = ensemble.predict_normal_params(
                x_eval.view(-1, 1)
            )

            results_per_ens_dict[ens_type]["pred_gammas"] = preds[:, :, 0]
            results_per_ens_dict[ens_type]["pred_nus"] = preds[:, :, 1]
            results_per_ens_dict[ens_type]["pred_alphas"] = preds[:, :, 2]
            results_per_ens_dict[ens_type]["pred_betas"] = preds[:, :, 3]

            results_per_ens_dict[ens_type]["mean_pred_mu"] = mean_mu
            results_per_ens_dict[ens_type]["mean_pred_sigma2"] = mean_sigma2

            # confidence bounds
            lower_mu, upper_mu = get_upper_lower_bounds_normal(
                p=0.975,
                mu=mean_mu.detach().numpy(),
                sigma=np.sqrt(var_mu.detach().numpy()),
            )

            lower_sigma2, upper_sigma2 = get_upper_lower_bounds_inv_gamma(
                p=0.975,
                alpha=mean_params[:, 2].detach().numpy(),
                beta=mean_params[:, 3].detach().numpy(),
            )

            results_per_ens_dict[ens_type]["lower_mu"] = lower_mu
            results_per_ens_dict[ens_type]["upper_mu"] = upper_mu
            results_per_ens_dict[ens_type]["lower_sigma"] = lower_sigma2
            results_per_ens_dict[ens_type]["upper_sigma"] = upper_sigma2

    # save results in a pickle file
    with open("results/" + exp_name + "/results_per_ens_dict.pkl", "wb") as f:
        pickle.dump(results_per_ens_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", dest="config_dir", type=str, required=True)
    args = parser.parse_args()
    _simulation_gamma_nig(config_dir=args.config_dir)