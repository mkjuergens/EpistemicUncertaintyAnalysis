import numpy as np
import torch


def entropy_dirichlet(alpha: torch.tensor):
    """calculates the entropy of a Dirichlet distribution with parameter vector alpha
    for each instance

    Parameters
    ----------
    alpha : torch.tensor of shape (n_instances, n_dim)
        tensor containing

    Returns
    -------
    torch.tensor of shape (n_instances, )
        (differential) entropy
    """

    alpha_0 = alpha.sum(dim=1)  # sum over all entries for each instance
    # calculate log of multivariate beta function for each instance parameters
    log_beta = torch.log(multiv_beta_fct(alpha))
    # calculate psigamma function on alpha_0
    psi_alpha_0 = torch.digamma(alpha_0)
    # calculate psi function on alpha
    psi_alpha = torch.digamma(alpha)
    # calculate entropy
    entropy = (
        log_beta
        - (alpha_0 - len(alpha)) * psi_alpha_0
        - ((alpha - 1) * psi_alpha).sum(dim=1)
    )

    return entropy


def entropy_regularizer_beta(params: list):
    """returns the entropy of the BEta distribution for each parameter pair in a pair of
      two lists of parameters alpha and beta.

    Parameters
    ----------
    params : list containing two arrays of shape (n_instances, n_dim)
        alpha and beta parameters of the Beta distribution per instance

    Returns
    -------
    np.array of shape (n_instances, n_dim)
        entropy per instance
    """

    alpha, beta = params # tensors of shape (n_instances, n_dim)
    # stack them to get a tensor of shape (n_instances, 2, n_dim)
    params = torch.stack((alpha, beta), dim=1)
    entropy = entropy_dirichlet(params)

    return entropy



def entropy_NIG(params: list):
    """returns the entropy of a Normal-Inverse-Gamma distribution given a tensor of parameters for
    each instance.

    Parameters
    ----------
    params : list
        list containing tensors of 4 parameters gamma, nu, alpha, beta defining the distribution for
        each instance

    Returns
    -------
    torch.tensor
        differential entropy per instance
    """
    gamma, nu, alpha, beta = params

    return (
        0.5
        + torch.log(np.sqrt(2 * np.pi) * (beta) ** (3 / 2))
        + torch.lgamma(alpha)
        - (torch.log(nu) / 2)
        + alpha
        - (alpha + 3 / 2) * torch.digamma(alpha)
    )


def multiv_beta_fct(alpha: torch.tensor):
    """apply the beta function to each row of alpha

    Parameters
    ----------
    alpha : torch.tensor
        tensor of shape (n_instances, n_dim)

    Returns
    -------
    torch.tensor
        tensor of shape (n_instances, )
    """

    return torch.exp(torch.lgamma(alpha).sum(dim=1) - torch.lgamma(alpha.sum(dim=1)))

def entropy_regularizer_nig(params: list):
    """computes sum of entropy for parameters of the normal-ionverse-gamma distribution used as 
    a regularization term in the second-order loss minimisastion. 

    Parameters
    ----------
    params : list
        list containing tensors of parameters gamma, nu, alpha, beta

    Returns
    -------
    float
        sum of instance-wise entropies
    """

    entropy = entropy_NIG(params)
    return entropy.sum()


if __name__ == "__main__":
    alpha = torch.tensor([[0.1, 0.1, 0.1], [4, 5, 6]], dtype=torch.float)
    print(multiv_beta_fct(alpha))
    print(entropy_dirichlet(alpha))
    params_nig = [
        torch.tensor([1, 20, 3]).reshape(3, 1),
        torch.tensor([0.1, 1, 0.1]).reshape(3, 1),
        torch.tensor([2, 2, 2]).reshape(3, 1),
        torch.tensor([2, 3, 10]).reshape(3,1),
    ]
    print(entropy_NIG(params_nig))
    params_beta = [torch.tensor([1, 1, 1, 4]).reshape(4,1), torch.tensor([2, 3, 4, 5]).reshape(4,1)]
    print(entropy_regularizer_beta(params=params_beta))
