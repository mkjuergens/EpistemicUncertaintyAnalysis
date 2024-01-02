import torch

from epuc.models import BetaNN, NIGNN, PredictorModel, RegressorModel
from epuc.losses import *

model_config = {
    "Bernoulli"  : {
        "model": PredictorModel,
        "kwargs": {
        "hidden_dim": 50,
        "output_dim": 1,
        "use_relu": False,
        "use_softplus": True,
        }
    },
    "Normal": {
        "model": RegressorModel,
        "kwargs": {
        "hidden_dim": 50,
        "use_softplus": True,
        "output_dim": 1,
        }

    },
    "Beta": {
        "model": BetaNN,
        "kwargs": {   
        "hidden_dim": 50,
        "use_softplus": True,
        "output_dim": 1,
        }

    },
    "NormalInverseGamma": {
        "model": NIGNN,
        "kwargs": {
        "hidden_dim": 50,
        "use_softplus": True,
        "output_dim": 1,
        }

    },   
}

train_config = {
    "Bernoulli": {
        "loss": torch.nn.BCELoss(),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Normal": {
        "loss": NegativeLogLikelihoodLoss(),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Beta_outer": {
        "loss": outer_bce_loss(lambda_reg=0.0),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Beta_outer_reg": {
        "loss": outer_bce_loss(lambda_reg=0.1),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Beta_inner": {
        "loss": inner_bce_loss(lambda_reg=0.0),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Beta_inner_reg": {
        "loss": inner_bce_loss(lambda_reg=0.1),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "NIG_outer": {
        "loss": outer_loss_der(lambda_reg=0.0),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "NIG_outer_reg": {
        "loss": outer_loss_der(lambda_reg=0.1, reg_type="evidence"),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
     "NIG_inner": {
        "loss": inner_loss_der(lambda_reg=0.0),
        "n_epochs": 5000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
     "NIG_inner_reg": {
        "loss": inner_loss_der(lambda_reg=0.1, reg_type="evidence"),
        "n_epochs": 2000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,

     }
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

