import torch 
import numpy as np

def evidence_regulizer_nig(params, y):
    """evindece bases regularization as proposed in Amini et al.

    Parameters
    ----------
    params : list
        parameters of the NIG distribution as predicted by the model.
    y : torch.tensor
        tensor of target valeus

    Returns
    -------
    float
        regularization loss
    """

    gamma, nu, alpha = params[:-1]

    abs_dif = torch.abs(y - gamma)

    reg = abs_dif * (2 * nu + alpha)

    return reg.sum()
