import torch
import numpy as np

from epuc.distances import sum_kl_divergence_nig

class NegativeLogLikelihoodLoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, params_normal: list, y):
        sigma = torch.clamp(params_normal[1], min=1e-3)
        mu = params_normal[0]

        loss = 0.5 * torch.log(2 * torch.pi * sigma**2) + (y - mu)**2 / (2 * sigma**2)

        return loss.mean()


class outer_loss_der(torch.nn.Module):

    def __init__(self, lambda_reg: float, epsilon: float = 0.0001):
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

    def __init__(self, lambda_reg: float, epsilon: float = 0.0001):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
    
    def forward(self, params_nig: list, y: torch.tensor):
        if self.lambda_reg > 0.0:
            sum_kl_div = sum_kl_divergence_nig(params_nig, epsilon=self.epsilon)
        else:
            sum_kl_div = 0.0

        inner_loss = inner_nig_loss(params_nig, y)

        return inner_loss + self.lambda_reg * sum_kl_div
    

def inner_nig_loss(params: list, y: torch.tensor):
    """secondary loss function using "inner expectation minimisation" and negative log likelihood as primary loss,
    assuming normally distributed data and a normal inverse gamma (NIG) distributed prior distribution.

    Parameters
    ----------
    params : list
        list of tensors of parameters of the NIG distribution per instance: [gamma, nu, alpha, beta]
    y : torch.tensor
        tensor of target values

    Returns
    -------
    log_loss    hybvimo;
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
    gamma, nu, alpha, beta = params[0], params[1], params[2], params[3]

    loss = (
        -0.5 * ((-alpha / beta) * (y - gamma) ** 2
        - 1 / nu
        + torch.digamma(alpha)
        - torch.log(beta)
        - torch.log(torch.ones(len(alpha)) * np.pi)
        )
    )

    return loss.sum()
