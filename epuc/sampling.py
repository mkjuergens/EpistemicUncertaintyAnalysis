import numpy as np
from scipy.stats import norm, invgamma


def sample_normal_params_from_normal_inverse_gamma(
    n_samples: int, gamma, nu, alpha, beta
):
    """
    Sample parameters of a normal distribution from a normal-inverse-gamma distribution.

    Parameters:
    n_samples: number of pairs of parameters sampled from the normal inverse gamma distribution
    gamma (float): Location parameter of the normal-inverse-gamma distribution.
    nu (float): Precision parameter of the normal-inverse-gamma distribution.
    alpha (float): Shape parameter of the inverse-gamma distribution.
    beta (float): Scale parameter of the inverse-gamma distribution.

    Returns:
    float, float: The sampled mean and standard deviation of the normal distribution.
    """
    # Sample from the inverse-gamma distribution to get the variance of the normal distribution
    taus = invgamma.rvs(a=alpha, scale=beta, size=n_samples)
    sigmas = np.sqrt(1.0 / taus)

    # Sample from the normal distribution to get the mean of the normal distribution
    means = norm.rvs(loc=gamma, scale=sigmas / np.sqrt(nu), size=n_samples)

    return means, sigmas


def sample_normal_params_from_nig_vectorised(
    n_samples: int, gammas, nus, alphas, betas
):
    """
    Sample multiple parameter pairs of a normal distribution for each instance from a normal-inverse-gamma distribution,
    where the parameters of the normal-inverse-gamma distribution are provided as arrays.

    Parameters:
    gammas (numpy.ndarray): Array of location parameters of the normal-inverse-gamma distribution.
    nus (numpy.ndarray): Array of precision parameters of the normal-inverse-gamma distribution.
    alphas (numpy.ndarray): Array of shape parameters of the inverse-gamma distribution.
    betas (numpy.ndarray): Array of scale parameters of the inverse-gamma distribution.
    n_samples_per_instance (int): Number of samples to generate per instance.

    Returns:
    List[numpy.ndarray]: List of arrays containing the sampled means for each instance.
    List[numpy.ndarray]: List of arrays containing the sampled standard deviations for each instance.
    """
    sampled_means = []
    sampled_sigmas = []

    for _ in range(n_samples):
        # Sample from the inverse-gamma distribution to get the variances for each instance
        taus = invgamma.rvs(a=alphas, scale=betas)
        sigmas = np.sqrt(1.0 / taus)

        # Sample from the normal distribution to get the means for each instance
        means = norm.rvs(loc=gammas, scale=sigmas / np.sqrt(nus))

        sampled_means.append(means)
        sampled_sigmas.append(sigmas)

    return sampled_means, sampled_sigmas


def sample_from_sampled_normal_distributions(sampled_means, sampled_sigmas, n_samples_per_distribution):
    """
    Sample values from the normal distributions defined by the sampled means and standard deviations.

    Parameters:
    sampled_means (List[numpy.ndarray]): List of arrays containing the sampled means for each resampling step.
    sampled_sigmas (List[numpy.ndarray]): List of arrays containing the sampled standard deviations for each resampling step.
    n_samples_per_distribution (int): Number of samples to generate from each normal distribution.

    Returns:
    List[List[numpy.ndarray]]: A list where each element is a list of arrays, each containing the sampled values from the normal distributions for each resampling step.
    """
    all_samples = []

    for means, sigmas in zip(sampled_means, sampled_sigmas):
        samples_per_step = []
        for mean, sigma in zip(means, sigmas):
            samples = norm.rvs(loc=mean, scale=sigma, size=n_samples_per_distribution)
            samples_per_step.append(samples)
        all_samples.append(samples_per_step)

    return all_samples
