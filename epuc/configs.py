import torch

from epuc.model import BetaNN, NIGNN, PredictorModel, RegressorModel
from epuc.losses import *

model_config = {
    "Bernoulli"  : {
        "model": PredictorModel,
        "kwargs": {
        "hidden_dim": 10,
        "output_dim": 1,
        "use_relu": False,
        "use_softplus": True,
        }
    },
    "Normal": {
        "model": RegressorModel,
        "kwargs": {
        "hidden_dim": 10,
        "use_softplus": True,
        "output_dim": 1,
        }

    },
    "Beta": {
        "model": BetaNN,
        "kwargs": {   
        "hidden_dim": 10,
        "use_softplus": True,
        "output_dim": 1,
        }

    },
    "NormalInverseGamma": {
        "model": NIGNN,
        "kwargs": {
        "hidden_dim": 10,
        "use_softplus": True,
        "output_dim": 1,
        }

    },   
}

train_config = {
    "Bernoulli": {
        "loss": torch.nn.BCELoss(),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Normal": {
        "loss": NegativeLogLikelihoodLoss(),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Beta_outer": {
        "loss": outer_bce_loss(lambda_reg=0.0),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Beta_outer_reg": {
        "loss": outer_bce_loss(lambda_reg=0.1),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Beta_inner": {
        "loss": inner_bce_loss(lambda_reg=0.0),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "Beta_inner_reg": {
        "loss": inner_bce_loss(lambda_reg=0.1),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "NIG_outer": {
        "loss": outer_loss_der(lambda_reg=0.0),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
    "NIG_outer_reg": {
        "loss": outer_loss_der(lambda_reg=0.1, reg_type="evidence"),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
     "NIG_inner": {
        "loss": inner_loss_der(lambda_reg=0.0),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,
    },
     "NIG_inner_reg": {
        "loss": inner_loss_der(lambda_reg=0.1, reg_type="evidence"),
        "n_epochs": 1000,
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.001},
        "batch_size": 128,
        "n_runs": 100,

     }
}

