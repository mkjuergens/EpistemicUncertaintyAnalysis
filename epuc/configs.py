import torch

from epuc.models import BetaNN, NIGNN, PredictorModel, RegressorModel
from epuc.helpers.ensemble import Ensemble, GaussianEnsemble, BetaEnsemble, NIGEnsemble
from epuc.losses import *


def create_train_config(
    type: str = "regression",
    lambda_reg: float = 0.1,
    n_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.001,
    reg_type: str = "evidence",
    ensemble_size: int = 100,
    ensemble_size_secondary: int = 1,
    hidden_dim: int = 50,
    n_hidden_layers: int = 1,
    output_dim: int = 1,
):
    """creates the training configuration for the specific problem given the training parameters.

    Parameters
    ----------
    type : str, optional
        type of problem, by default "regression". Options are {"regression", "classification"}
    lambda_reg : float, optional
        regularization strength, by default 0.1
    n_epochs : int, optional
        number of epochs , by default 1000
    batch_size : int, optional
        batch size, by default 128
    lr : float, optional
        learning rate, by default 0.001
    reg_type : str, optional
        regularization type (used in regression losses), by default "evidence". Options are
          {"evidence", "kl", "entropy"}
    ensemble_size : int, optional
        ensemble size, by default 100
    ensemble_size_secondary : int, optional
        ensemble size for the secondary distribution, by default 1

    Returns
    -------
    dict
        dictionary with parameters for each experiment

    """
    if type == "regression":
        return create_train_config_regression(
            lambda_reg=lambda_reg,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            reg_type=reg_type,
            ensemble_size=ensemble_size,
            ensemble_size_secondary=ensemble_size_secondary,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
        )
    elif type == "classification":
        return create_train_config_classification(
            lambda_reg=lambda_reg,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            ensemble_size=ensemble_size,
            ensemble_size_secondary=ensemble_size_secondary,
            reg_type=reg_type,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
        )
    else:
        raise NotImplementedError


def create_model_config(
    hidden_dim: int = 50,
    n_hidden_layers: int = 1,
    output_dim: int = 1,
    use_softplus: bool = True,
):
    """creates the model configuration for the specific problem given the model parameters.

    Parameters
    ----------
    type : str, optional
        type of problem, by default "regression". Options are {"regression", "classification"}
    hidden_dim : int, optional
        number of hidden units, by default 50
    n_hidden_layers : int, optional
        number of hidden layers, by default 1
    output_dim : int, optional
        output dimension, by default 1
    use_softplus : bool, optional
        whether to use softplus activation function, by default True

    Returns
    -------
    dict
        dictionary with parameters for each experiment

    """
    model_config = {
        "Bernoulli": {
            "model": PredictorModel,
            "kwargs": {
                "hidden_dim": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
                "output_dim": output_dim,
                "use_softplus": use_softplus,
            },
        },
        "Normal": {
            "model": RegressorModel,
            "kwargs": {
                "hidden_dim": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
                "use_softplus": use_softplus,
                "output_dim": output_dim,
            },
        },
        "Beta": {
            "model": BetaNN,
            "kwargs": {
                "hidden_dim": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
                "use_softplus": use_softplus,
                "output_dim": output_dim,
            },
        },
        "NormalInverseGamma": {
            "model": NIGNN,
            "kwargs": {
                "hidden_dim": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
                "use_softplus": use_softplus,
                "output_dim": output_dim,
            },
        },
    }
    return model_config


model_config = {
    "Bernoulli": {
        "model": PredictorModel,
        "kwargs": {
            "hidden_dim": 50,
            "n_hidden_layers": 1,
            "output_dim": 1,
            "use_softplus": True,
        },
    },
    "Normal": {
        "model": RegressorModel,
        "kwargs": {
            "hidden_dim": 50,
            "n_hidden_layers": 1,
            "use_softplus": True,
            "output_dim": 1,
        },
    },
    "Beta": {
        "model": BetaNN,
        "kwargs": {
            "hidden_dim": 50,
            "n_hidden_layers": 1,
            "use_softplus": True,
            "output_dim": 1,
        },
    },
    "NormalInverseGamma": {
        "model": NIGNN,
        "kwargs": {
            "hidden_dim": 50,
            "n_hidden_layers": 1,
            "use_softplus": True,
            "output_dim": 1,
        },
    },
}

data_config = {
    "classification": {
        "sine": {
            "n_samples_1": 500,
            "n_samples_2": 0,
            "x_min": 0.0,
            "x_max": 1.0,
            "x_split": 0.5,
            "sine_factor": 2.0,
            "amplitude": 0.8,
        },
        "linear": {
            "n_samples_1": 500,
            "n_samples_2": 0,
            "x_min": 0.0,
            "x_max": 1.0,
            "x_split": 0.5,
            "slope": -1.0,
            "intercept": 1.0,
        },
    },
    "regression": {
        "sine": {
            "n_samples_1": 500,
            "n_samples_2": 10,
            "x_min": 0.0,
            "x_max": 1.0,
            "x_split": 0.5,
            "eps_std": 0.03,
            "sine_factor": 2.0,
        },
        "polynomial": {
            "n_samples": 500,
            "x_min": -4,
            "x_max": 4,
            "degree": 3,
            "eps_std": 3,
        },
    },
}


def create_train_config_regression(
    lambda_reg: float = 0.1,
    n_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.001,
    reg_type: str = "evidence",
    ensemble_size: int = 100,
    ensemble_size_secondary: int = 1,
    hidden_dim: int = 50,
    n_hidden_layers: int = 1,
    output_dim: int = 1,
):
    """
    creates the cinfiguration for all models trained for the regression experiments
    """
    model_config = create_model_config(
        hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, output_dim=output_dim
    )
    train_config_regression = {
        "Normal": {
            "model_config": model_config["Normal"],
            "ensemble": GaussianEnsemble,
            "loss": NegativeLogLikelihoodLoss(),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size,
        },
        "NIG_outer": {
            "model_config": model_config["NormalInverseGamma"],
            "ensemble": NIGEnsemble,
            "loss": outer_loss_der(lambda_reg=0.0),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size_secondary,
        },
        "NIG_outer_reg": {
            "model_config": model_config["NormalInverseGamma"],
            "ensemble": NIGEnsemble,
            "loss": outer_loss_der(lambda_reg=0.1, reg_type=reg_type),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size_secondary,
        },
        "NIG_inner": {
            "model_config": model_config["NormalInverseGamma"],
            "ensemble": NIGEnsemble,
            "loss": inner_loss_der(lambda_reg=0.0),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size_secondary,
        },
        "NIG_inner_reg": {
            "model": NIGNN,
            "ensemble": NIGEnsemble,
            "loss": inner_loss_der(lambda_reg=lambda_reg, reg_type=reg_type),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size_secondary,
        },
    }
    return train_config_regression


def create_train_config_classification(
    lambda_reg: float = 0.1,
    n_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.001,
    reg_type: str = "kl",
    ensemble_size: int = 100,
    ensemble_size_secondary: int = 1,
    hidden_dim: int = 50,
    n_hidden_layers: int = 1,
    output_dim: int = 1,
):
    model_config = create_model_config(
        hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, output_dim=output_dim
    )
    train_config_classification = {
        "Bernoulli": {
            "model_config": model_config["Bernoulli"],
            "ensemble": Ensemble,
            "loss": torch.nn.BCELoss(),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size,
        },
        "Beta_outer": {
            "model_config": model_config["Beta"],
            "ensemble": BetaEnsemble,
            "loss": outer_bce_loss(lambda_reg=0.0),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size_secondary,
        },
        "Beta_outer_reg": {
            "model_config": model_config["Beta"],
            "ensemble": BetaEnsemble,
            "loss": outer_bce_loss(lambda_reg=lambda_reg, reg_type=reg_type),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size_secondary,
        },
        "Beta_inner": {
            "model_config": model_config["Beta"],
            "ensemble": BetaEnsemble,
            "loss": inner_bce_loss(lambda_reg=0.0),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size_secondary,
        },
        "Beta_inner_reg": {
            "model_config": model_config["Beta"],
            "ensemble": BetaEnsemble,
            "loss": inner_bce_loss(lambda_reg=0.1, reg_type=reg_type),
            "n_epochs": n_epochs,
            "optim": torch.optim.Adam,
            "optim_kwargs": {"lr": lr},
            "batch_size": batch_size,
            "ensemble_size": ensemble_size_secondary,
        },
    }

    return train_config_classification
