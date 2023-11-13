import torch
import numpy as np

from epuc.distances import sum_kl_divergence_nig


class NegativeLogLikelihoodLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, params_normal: list, y):
        sigma = torch.clamp(params_normal[1], min=1e-3)
        mu = params_normal[0]

        loss = 0.5 * torch.log(2 * torch.pi * sigma**2) + (y - mu) ** 2 / (
            2 * sigma**2
        )

        return loss.mean()


class outer_loss_der(torch.nn.Module):
    """outer expectation minimisation for the NIG-prior case.
    """
    def __init__(self, lambda_reg: float, type_reg: str = "evidence", epsilon: float = 0.0001):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon

    def forward(self, params_nig: list, y: torch.tensor):
        if self.lambda_reg > 0.0:
            sum_kl_div = sum_kl_divergence_nig(params_nig, epsilon=self.epsilon)
        else:
            sum_kl_div = 0.0

        outer_loss = outer_nig_loss(params_nig, y)

        return outer_loss + self.lambda_reg * sum_kl_div


class inner_loss_der(torch.nn.Module):
    """
    inner expectation minimization for the NIG-prior case. It optiimzes the parameters
    of the Normal-Inverse-Gamma distribution which yields estimates of the aleatoric and epsitemic
    uncertainty as well as predictions, as described in Amini et al.
    """

    def __init__(
        self, lambda_reg: float, reg_type: str = "kl", epsilon: float = 0.0001
    ):
        """
        Parameters
        ----------
        lambda_reg : float
           weight factor of regulaization
        reg_type : string
            type of regularization used. Must be in {"kl", "evidence"}
        epsilon : float, optional
            _description_, by default 0.0001
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.reg_type = reg_type

    def forward(self, params_nig: list, y: torch.tensor):
        if self.lambda_reg > 0.0:
            if self.reg_type == "kl":
                reg = sum_kl_divergence_nig(params_nig, epsilon=self.epsilon)
            elif self.reg_type == "evidence":
                reg = evidence_regulizer(params_nig, y)
            else:
                raise TypeError
        else:
            reg = 0.0

        inner_loss = inner_nig_loss(params_nig, y)

        return inner_loss + self.lambda_reg * reg


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

    return log_loss.sum()


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

    return loss.sum()


def evidence_regulizer(params, y):
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
