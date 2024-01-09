import torch
import numpy as np

from epuc.regularization import (
    reg_kl_div_nig,
    entropy_NIG,
    evidence_regulizer_nig,
)


class NegativeLogLikelihoodLoss(torch.nn.Module):
    """
    negative log likelihood loss for the Gaussian distribution.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, params_normal: list, y):
        sigma = torch.clamp(params_normal[1], min=1e-3)
        mu = params_normal[0]

        loss = 0.5 * torch.log(2 * torch.pi * sigma**2) + (y - mu) ** 2 / (
            2 * sigma**2
        )

        return loss.mean()


def get_reg_val(params, y, reg_type: str = "evidence"):
    if reg_type == "kl":
        reg = reg_kl_div_nig(params, epsilon=0.0001)
    elif reg_type == "evidence":
        reg = evidence_regulizer_nig(params, y)
    elif reg_type == "entropy":
        # take negative of entropy (entropy is to be maximised)
        reg = - entropy_NIG(params)
    else:
        raise NotImplementedError
    return reg


class outer_loss_der(torch.nn.Module):
    """
    outer expectation minimisation for the NIG-prior case.
    """

    def __init__(self, lambda_reg: float, reg_type: str = "evidence"):
        """
        Parameters
        ----------
        lambda_reg : float
            weight coefficient for the regularization parameter
        reg_type : str, optional
            type of regularization used, by default "evidence". other options are {"kl", "entropy"}
        epsilon : float, optional
            parameter adeed enabling computation of KL divergence, by default 0.0001
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type

    def forward(self, params_nig: list, y: torch.tensor):
        if self.lambda_reg > 0.0:
            reg = get_reg_val(params=params_nig, y=y, reg_type=self.reg_type)
        else:
            reg = 0.0

        outer_loss = outer_nig_loss(params_nig, y)
        loss = outer_loss + self.lambda_reg * reg

        return loss.mean()


class inner_loss_der(torch.nn.Module):
    """
    inner expectation minimization for the NIG-prior case. It optiimzes the parameters
    of the Normal-wInverse-Gamma distribution which yields estimates of the aleatoric and epsitemic
    uncertainty as well as predictions, as described in Amini et al.
    """

    def __init__(
        self, lambda_reg: float, reg_type: str = "evidence", epsilon: float = 0.0001
    ):
        """
        Parameters
        ----------
        lambda_reg : float
           weight factor of regulaization
        reg_type : string
            type of regularization used. Must be in {"kl", "evidence", "entropy"}
        epsilon : float, optional
            stability parameter, by default 0.0001
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.reg_type = reg_type

    def forward(self, params_nig: list, y: torch.tensor):
        if self.lambda_reg > 0.0:
            reg = get_reg_val(params=params_nig, y=y, reg_type=self.reg_type)
        else:
            reg = 0.0

        inner_loss = inner_nig_loss(params_nig, y)
        loss = inner_loss + self.lambda_reg * reg

        return loss.mean()


def inner_nig_loss(params: list, y: torch.tensor):
    """
    secondary loss function using "inner expectation minimisation" and negative log likelihood as primary loss,
    assuming normally distributed data and a normal inverse gamma (NIG) distributed prior distribution.

    Parameters
    ----------
    params : list
        list of tensors of parameters of the NIG distribution per instance: [gamma, nu, alpha, beta]
    y : torch.tensor
        tensor of target values

    Returns
    -------
    float
        loss
    """
    gamma, nu, alpha, beta = params

    omega = 2 * beta * (1 + nu)
    # log terms
    log_1 = torch.log(torch.ones(len(nu)) * np.pi) - torch.log(nu)
    log_2 = torch.log(omega)
    log_3 = torch.log(((y - gamma) ** 2) * nu + omega)
    log_4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    # now put it all together
    log_loss = 0.5 * log_1 - alpha * log_2 + (alpha + 0.5) * log_3 + log_4

    return log_loss


def outer_nig_loss(params: list, y: torch.tensor):
    """secodnary loss function using "outer expectaiton minimisation" and negative log likelihood
    as primary loss, assuming normally distributed data and a normal inverse gamma (NIG) distribtued
    prior distribution.

    Parameters
    ----------
    params : list
        list of (tensors of) parameters of the NIG distribution per instance: [gamma, nu, alpha, beta]
    y : torch.tensor
        tensor of target values

    Returns
    -------
    float
        loss
    """
    gamma, nu, alpha, beta = params

    loss = -0.5 * (
        (-alpha / beta) * (y - gamma) ** 2
        - 1 / nu
        + torch.digamma(alpha)
        - torch.log(beta)
        - torch.log(torch.ones(len(alpha)) * np.pi)
    )

    return loss
