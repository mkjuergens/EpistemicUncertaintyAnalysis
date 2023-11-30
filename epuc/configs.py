import torch

from epuc.model import BetaNN, NIGNN, PredictorModel, RegressorModel
from epuc

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
        "loss"

