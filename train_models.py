import json
import os
import argparse
from typing import Optional
import datetime
import pickle

import numpy as np

from epuc.datasets import create_evaluation_data

from epuc.configs import create_train_config, create_data_config

from epuc.helpers.parameter_analysis import compute_upper_lower_bounds_ensemble

def _main_training(
    config_dir,
    n_samples: int,
    ens_type: Optional[str] = None,
    type: str = "regression",
    data_type: str = "polynomial",
    exp_name: Optional[list] = None,
    save_dir: str = "results",
    resample_train_data: bool = True,
    range_x_eval: Optional[tuple] = None,
):
    """function for training one or multiple ensemble models, saving the trained models 
    in pkl files.

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
    resturn_losses: bool
        whether to return the losses per training epoch (of each ensemble member)
    resample_train_data : bool, optional
        whether to resample the training data for each ensemble member, by default True
    plot_results : bool, optional
        whether to plot the results, by default True
    """

    if not exp_name:
        exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # load json file located in the config_dir directory into a dictionary
    with open(config_dir) as json_file:
        temp_dict = json.load(json_file)
        train_config = create_train_config(type=type, **temp_dict)
        temp_dict["n_samples"] = n_samples

    save_path = f"{save_dir}/" + type + f"/{exp_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "/params.json", "w") as outfile:
        json.dump(temp_dict, outfile)

    data_config = create_data_config(n_samples=n_samples)

    dataset, x_eval, y_eval, x_train, y_targets = create_evaluation_data(
        data_config=data_config,
        problem_type=type,
        data_type=data_type,
        n_eval_points=1000,
        range_x_eval=range_x_eval,
    )

    if ens_type:
        keys = list(ens_type)
    else:
        keys = train_config.keys()
    for ens_type in keys:

        ensemble = train_config[ens_type]["ensemble"](
            model_config=train_config[ens_type]["model_config"],
            ensemble_size=train_config[ens_type]["ensemble_size"],
        )
        ensemble.train(
            dataset=dataset,
            data_params=data_config[type][data_type],
            train_params=train_config[ens_type],

            resample_train_data=resample_train_data,
            x_eval=x_eval,
        )
        save_path_model = save_path + f"/{ens_type}.pkl"
        if not os.path.exists(save_path_model):
            os.makedirs(save_path_model)
        ensemble.save_models(save_path_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("--config_dir", dest="config_dir", type=str, required=True)
    parser.add_argument("--data_type", dest="data_type", type=str, required=True)
    # optional arguments
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="results")
    parser.add_argument("--type", dest="type", type=str, default="regression")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--n_samples", dest="n_samples", type=int, default=1000)
    parser.add_argument(
        "--resample_train_data", dest="resample_train_data", type=bool, default=True
    )
    parser.add_argument(
        "--range_x_eval", nargs="+", type=float, dest="range_x_eval",  default=None
    )

    args = parser.parse_args()

    _main_training(
        config_dir=args.config_dir,
        n_samples=args.n_samples,
        type=args.type,
        data_type=args.data_type,
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        resample_train_data=args.resample_train_data,
        range_x_eval=args.range_x_eval,
    )    
