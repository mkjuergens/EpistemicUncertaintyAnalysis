import torch

from epuc.models import BetaNN, NIGNN, PredictorModel, RegressorModel
from epuc.losses import *

def create_train_config(type: str = "regression", lambda_reg: float = 0.1, n_epochs: int = 1000,
                        batch_size: int = 128, lr: float = 0.001, reg_type: str = "evidence"):
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

    Returns
    -------
    dict
        dictionary with parameters for each experiment

    """
    if type == "regression":
        return create_train_config_regression(lambda_reg=lambda_reg, n_epochs=n_epochs,
                                              batch_size=batch_size, lr=lr, reg_type=reg_type)
    elif type == "classification":
        return create_train_config_classification(lambda_reg=lambda_reg, n_epochs=n_epochs,
                                                  batch_size=batch_size, lr=lr)
    else:
        raise NotImplementedError
    


model_config = {
    "Bernoulli"  : {
        "model": PredictorModel,
        "kwargs": {
        "hidden_dim": 50,
        "n_hidden_layers": 1,
        "output_dim": 1,
        "use_relu": False,
        "use_softplus": True,
        }
    },
    "Normal": {
        "model": RegressorModel,
        "kwargs": {
        "hidden_dim": 50,
        "n_hidden_layers": 1,
        "use_softplus": True,
        "output_dim": 1,
        }

    },
    "Beta": {
        "model": BetaNN,
        "kwargs": {   
        "hidden_dim": 50,
        "n_hidden_layers": 1,
        "use_softplus": True,
        "output_dim": 1,
        }

    },
    "NormalInverseGamma": {
        "model": NIGNN,
        "kwargs": {
        "hidden_dim": 50,
        "n_hidden_layers": 1,
        "use_softplus": True,
        "output_dim": 1,
        }

    },   
}

data_config = {
    "BernoulliSine": {
        "n_samples_1": 5000,
        "n_samples_2": 10,
        "x_min": 0.0,
        "x_max": 1.0,
        "x_split": 0.5,
        "sine_factor": 5.0
    },
    "SineRegression": {
        "n_samples_1": 1000,
        "n_samples_2": 10,
        "x_min": 0.0,
        "x_max": 1.0,
        "x_split": 0.5,
        "sine_factor": 5.0,
        "eps_std": 0.03
    },
    "PolynomialRegression": {
        "n_samples": 1000,
        "x_min": -4,
        "x_max": 4,
        "degree": 3,
        "eps_std": 3
    }
    
}

def create_train_config_regression(lambda_reg: float = 0.1, n_epochs: int = 1000, 
                                    batch_size: int = 128, lr: float = 0.001,
                                    reg_type: str = "evidence"):
    """creates the cinfiguration for all models trained for the regression experiments
    """
    train_config_regression = {
    "Normal": {
        "loss": NegativeLogLikelihoodLoss,
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,
    },
    "NIG_outer": {
        "loss": outer_loss_der(lambda_reg=0.0),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,
    },
    "NIG_outer_reg": {
        "loss": outer_loss_der(lambda_reg=0.1, reg_type=reg_type),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,
    },
    "NIG_inner": {
        "loss": inner_loss_der(lambda_reg=0.0),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,
    },
    "NIG_inner_reg": {
        "loss": inner_loss_der(lambda_reg=lambda_reg, reg_type=reg_type),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,

            }
        }    
    return train_config_regression
                    

def create_train_config_classification(lambda_reg: float = 0.1, n_epochs: int = 1000,
                                       batch_size: int = 128, lr: float = 0.001):
    
    train_config_classification = {
    "Bernoulli": {
        "loss": torch.nn.BCELoss(),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,
    },
    "Beta_outer": {
        "loss": outer_bce_loss(lambda_reg=0.0),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,
    },
    "Beta_outer_reg": {
        "loss": outer_bce_loss(lambda_reg=lambda_reg),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size":batch_size,
    },
    "Beta_inner": {
        "loss": inner_bce_loss(lambda_reg=0.0),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,
    },
    "Beta_inner_reg": {
        "loss": inner_bce_loss(lambda_reg=0.1),
        "n_epochs": n_epochs,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": lr},
        "batch_size": batch_size,
    },
        }
    
    return train_config_classification