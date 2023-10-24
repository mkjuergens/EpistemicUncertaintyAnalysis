import numpy as np
import torch
from scipy.special import gammaln
import scipy.special as sps

def kl_divergence_dirichlet(alpha, beta):
    """
    Compute the KL divergence between two Dirichlet distributions P and Q,
    with parameters alpha and beta respectively.
    
    Parameters:
        alpha (array_like): Parameters of distribution P.
        beta (array_like): Parameters of distribution Q.
    
    Returns:
        float: KL divergence between P and Q.
    """
    # Ensure the parameters are Numpy arrays
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    
    # Check if the arrays are of the same shape
    if alpha.shape != beta.shape:
        raise ValueError("The parameter arrays alpha and beta must have the same shape.")
    
    # Compute the log-gamma and digamma functions
    log_gamma_alpha = sps.gammaln(alpha)
    log_gamma_beta = sps.gammaln(beta)
    log_gamma_alpha_sum = sps.gammaln(np.sum(alpha))
    log_gamma_beta_sum = sps.gammaln(np.sum(beta))
    psi_alpha = sps.digamma(alpha)
    psi_alpha_sum = sps.digamma(np.sum(alpha))
    
    # KL divergence computation using the formula
    kl = (log_gamma_alpha_sum - log_gamma_beta_sum + 
          np.sum(log_gamma_beta - log_gamma_alpha + 
                 (alpha - beta) * (psi_alpha - psi_alpha_sum)))
    
    return kl

def kl_divergence_dirichlet_torch(alpha_1, alpha_2):
    """
    Compute the KL divergence between two Dirichlet distributions.
    
    Parameters:
    - alpha_1: Parameter vector for the first Dirichlet distribution
    - alpha_2: Parameter vector for the second Dirichlet distribution
    
    Returns:
    - KL divergence between the two Dirichlet distributions
    """
    
    # Gamma function in PyTorch is the factorial for integers, but we can use the lgamma (log-gamma) function directly
    assert alpha_1.shape == alpha_2.shape, "alpha_1 and alpha_2 must have the same shape"
    sum_alpha_1 = torch.sum(alpha_1)
    sum_alpha_2 = torch.sum(alpha_2)
    
    kl = torch.lgamma(sum_alpha_1) - torch.lgamma(sum_alpha_2) - torch.sum(torch.lgamma(alpha_1)) + torch.sum(torch.lgamma(alpha_2))
    
    kl += torch.sum((alpha_1 - alpha_2) * (torch.digamma(alpha_1) - torch.digamma(sum_alpha_1)))
    
    return kl



if __name__ == "__main__":
    alpha_1 = torch.tensor([0.1, 0.1])
    alpha_2 = torch.ones(2)

    print(kl_divergence_dirichlet_torch(alpha_2, alpha_1))