import numpy as np
import torch
import scipy.special as sps


def kl_divergence_dirichlet(alpha, beta):
    """
    Compute the KL divergence between two Dirichlet distributions P and Q,
    with parameters alpha and beta respectively.

    Parameters:
    ----------------
        alpha (array_like): Parameters of distribution P.
        beta (array_like): Parameters of distribution Q.

    Returns:
    ----------------
        float: KL divergence between P and Q.
    """
    # Ensure the parameters are Numpy arrays
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    # Check if the arrays are of the same shape
    if alpha.shape != beta.shape:
        raise ValueError(
            "The parameter arrays alpha and beta must have the same shape."
        )

    # Compute the log-gamma and digamma functions
    log_gamma_alpha = sps.gammaln(alpha)
    log_gamma_beta = sps.gammaln(beta)
    log_gamma_alpha_sum = sps.gammaln(np.sum(alpha))
    log_gamma_beta_sum = sps.gammaln(np.sum(beta))
    psi_alpha = sps.digamma(alpha)
    psi_alpha_sum = sps.digamma(np.sum(alpha))

    # KL divergence computation using the formula
    kl = (
        log_gamma_alpha_sum
        - log_gamma_beta_sum
        + np.sum(
            log_gamma_beta
            - log_gamma_alpha
            + (alpha - beta) * (psi_alpha - psi_alpha_sum)
        )
    )

    return kl


def kl_divergence_dirichlet_torch(alpha_1, alpha_2):
    """
    Compute the KL divergence between two Dirichlet distributions.

    Parameters:
    ----------------
        alpha_1: Parameter vector for the first Dirichlet distribution
        alpha_2: Parameter vector for the second Dirichlet distribution

    Returns:
    ----------------
        float:
            KL divergence between the two Dirichlet distributions
    """

    # Gamma function in PyTorch is the factorial for integers, but we can use the lgamma (log-gamma) function directly
    assert (
        alpha_1.shape == alpha_2.shape
    ), "alpha_1 and alpha_2 must have the same shape"
    sum_alpha_1 = torch.sum(alpha_1)
    sum_alpha_2 = torch.sum(alpha_2)

    kl = (
        torch.lgamma(sum_alpha_1)
        - torch.lgamma(sum_alpha_2)
        - torch.sum(torch.lgamma(alpha_1))
        + torch.sum(torch.lgamma(alpha_2))
    )

    kl += torch.sum(
        (alpha_1 - alpha_2) * (torch.digamma(alpha_1) - torch.digamma(sum_alpha_1))
    )

    return kl


def kl_divergergence_nig_torch(nu, alpha, epsilon: float = 0.001):
    """calculates the KL-divergence between an inverse digamma distribution with second and third parameters
    nu and alpha, respectively, and the (approximate zero evidnece) KL-divergence (with second parameter epsilon,
    third parameter 1 + epsilon).

    Parameters
    ----------
    nu : torch.tensor
        second parameter of the digamma distribution per instance
    alpha : torch.tensor
        third parameter of the digamma distribution per instance
    epsilon : float, optional
        parameter which allows computing the KL divergence, by default 0.001

    Returns
    -------
    torch.tensor
        kl divergence per instance
    """
    eps_tensor = torch.zeros(len(nu)) + epsilon
    psi_alpha = torch.digamma(alpha)
    log_gamma_alpha = torch.lgamma(alpha)
    log_gamma_eps = torch.lgamma(1 + eps_tensor)

    kl = (
        0.5 * (1 + eps_tensor) / nu
        - 0.5
        - log_gamma_alpha
        + log_gamma_eps
        + (alpha - (1 + eps_tensor)) * psi_alpha
    )

    return kl


def kl_divergence_beta(beta_params: list):
    """calculate the sum of the KL-divergenses between (multiple) parameters of the Beta distribution
    and the Beta(1,1) distribution (i.e. the uniform distribution over the simplex) for multiple instances.

    Parameters
    ----------
    beta_params : list
        list containing two tensors which are the first and second parameter of the induced
        Beta distribution

    Returns
    -------
    float
        average KL-divergence
    """

    alpha, beta = beta_params
    device = alpha.get_device() if alpha.get_device() >= 0 else "cpu"
    assert alpha.shape == beta.shape, "alpha and beta must have the same shape"
    beta_p = torch.zeros(alpha.shape[0], 2).to(device)
    beta_p[:, 0] = alpha.squeeze()
    beta_p[:, 1] = beta.squeeze()

    # calculate kl divergence to uniform dirichlet distribution (i.e. Dir([1,1,...,1]))
    # for every parameter pair
    kl_div = torch.stack(
        [
            kl_divergence_dirichlet_torch(x_i, torch.ones(2).to(device))
            for x_i in torch.unbind(beta_p, dim=0)
        ],
        dim=0,
    )
    # calculate sum
    # expected_kl_div = torch.sum(kl_div) # do not take sum anymore as we average in the end

    return kl_div


def reg_kl_div_nig(params_nig: list, epsilon: float = 0.0001):
    nu, alpha = params_nig[1], params_nig[2]
    assert (
        nu.shape == alpha.shape
    ), "parameters of the normal-inverse-gamma distribution should have the same shape"

    kl_div = kl_divergergence_nig_torch(
        nu.squeeze(), alpha.squeeze(), epsilon=epsilon
    )  # tensor with value per instance

    return kl_div


if __name__ == "__main__":
    alpha_1 = torch.tensor([0.1, 0.1])
    alpha_2 = torch.ones(2)

    nu = torch.zeros(10, 1) + 0.1
    alpha = torch.zeros(10, 1) + 0.5
    gamma = torch.zeros(10, 1) + 0.1
    beta = torch.zeros(10, 1) + 0.1
    params = [gamma, nu, alpha, beta]

    # now test kl divergence for normal inverse gamma distribution
    nu = torch.tensor([0.1, 0.3])
    alpha = torch.tensor([0.1, 0.1])

    print(kl_divergence_dirichlet_torch(alpha_2, alpha_1))
    print(kl_divergergence_nig_torch(nu, alpha))
    print(torch.sum(kl_divergergence_nig_torch(nu, alpha)))
    print(reg_kl_div_nig(params))
