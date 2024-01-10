import torch

from epuc.regularization.distances import sum_kl_divergence_beta


class outer_bce_loss(torch.nn.Module):
    """
    Class for the outer loss function for the second order loss. 
    """

    def __init__(self, lambda_reg: float, estimate: str = "analytical", n_samples: int = 500):
        """
        Parameters
        ----------
        lambda_reg : float
            weighting factor of the regularisation term
        estimate : string
            way to calculate loss function. Needs to be in {"mcmc", "analytical"}.
        n_samples : int, optional
            number of samples generated to estimate the expectation , by default 100
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.n_samples = n_samples
        self.estimate = estimate

    def forward(self, params_beta: list, y: torch.tensor):
        """forward pass of the outer secondary loss minimisation using a BEta prior and the 
        binary cross entropy loss as the primary loss function.

        Parameters
        ----------
        params_dir : list
            list containing wo tensors which are the first and second parameter of the 
            Beta distribution   
        y : torch.tensor
            binary labels
        """

        if self.lambda_reg > 0.0:
            expected_kl_div = sum_kl_divergence_beta(params_beta)
        else:
            expected_kl_div = 0.0
        if self.estimate == "analytical":
            # calculate expected secondary loss 
            expected_bce = outer_bce_loss_beta(params_beta, y)
        elif self.estimate == "mcmc":
        # caclculate expected secondary outer expectation loss
            expected_bce = expected_outer_bce_loss_mcmc(params_beta, y, n_samples=self.n_samples)
        else:
            raise NotImplementedError

        return expected_bce + self.lambda_reg * expected_kl_div
    

class inner_bce_loss(torch.nn.Module):
    """
    Class for the inner loss minimization for the second order loss. 
    """

    def __init__(self, lambda_reg: float, n_samples: int = 500, estimate="analytical"):
        """
        Parameters
        ----------
        lambda_reg : float
            weighting factor of the regularisation term
        estimate : string
            way to calculate loss function. Needs to be in {"mcmc", "analytical"}.
        n_samples : int, optional
            number of samples generated to estimate the expectation , by default 100
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.n_samples = n_samples
        self.estimate = estimate

    def forward(self, params_beta: list, y: torch.tensor):
        
        if self.lambda_reg > 0.0:
            expected_kl_div = sum_kl_divergence_beta(params_beta)
        else:
            expected_kl_div = 0.0

        if self.estimate == "mcmc":
            expected_inner_bce = inner_bce_loss_mcmc(params_beta, y, n_samples=self.n_samples)
        elif self.estimate == "analytical":
            expected_inner_bce = inner_bce_loss_beta(params_beta, y)
        else:
            raise NotImplementedError
        
        return expected_inner_bce + self.lambda_reg * expected_kl_div


def expected_outer_bce_loss_mcmc(params: list, y: torch.tensor, n_samples: int = 100):
    """calculates an approximation of the second order loss function using (leaned) parameters 
    (alpha, beta) of a Beta distribution

    Parameters
    ----------
    y : torch.tensor
        labels 
    params: list of torch.tensor
        list containing the parameters of the Beta distribution (i.e. alpha and beta)
    n_samples : int, optional
        number of samples to be used to aprximate the mean, by default 100

    Returns
    -------
    float
    """

    # sample theta from Beta(alpha, beta)
    alpha, beta = params[0], params[1]
    theta_samples = torch.distributions.beta.Beta(alpha, beta).rsample([n_samples])
    # compute binary cross entropy loss for each sample
    bce_losses = - y * torch.log(theta_samples) - (1 - y) * torch.log(1 - theta_samples)
    # average over the samples to approximate the expectation
    expected_bce = bce_losses.mean(dim=0)

    return expected_bce.mean()

def outer_bce_loss_beta(params: list, y: torch.tensor):
    alphas, betas = params[0], params[1]

    dig_a = torch.digamma(alphas)
    dig_b = torch.digamma(betas)
    dig_sum = torch.digamma(alphas + betas)

    bce_losses = - y * (dig_a - dig_sum) - (1 - y) * (dig_b - dig_sum)

    return bce_losses.sum()


def inner_bce_loss_beta(params: list, y: torch.tensor):

    alpha, beta = params[0], params[1]
    assert alpha.shape == beta.shape, "alpha and beta must have the same shape"
    assert len(alpha) == len(y), "must have same number of prior parameters as labels"

    losses = posterior_categ_beta(alpha, beta, y)
    return losses.sum()    

def inner_bce_loss_mcmc(params: list, y: torch.tensor, n_samples: int = 100):

    alpha, beta = params[0], params[1]

    theta_samples = torch.distributions.beta.Beta(alpha, beta).rsample([n_samples])

    bce_losses = - y * torch.log(theta_samples) - (1 - y) * torch.log(1 - theta_samples)

    return bce_losses.mean()

def posterior_categ_beta(alpha: torch.tensor, beta: torch.tensor, y: torch.tensor):
    losses = - y * torch.log(alpha / (alpha + beta)) - (1 - y) * torch.log(beta / (alpha + beta))
    return losses

if __name__ == "__main__":
    alpha = torch.ones(100, 1)*0.1
    beta = torch.ones(100, 1)*0.1
    params = [alpha, beta]
    y = torch.ones(100, 1)
    loss_outer = outer_bce_loss(lambda_reg=1.0)
    print(loss_outer(params, y))
    print(inner_bce_loss_beta(params, y))
    print(outer_bce_loss_beta(params, y))

