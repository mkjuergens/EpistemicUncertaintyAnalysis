import torch
import numpy as np
from scipy.stats import norm

def normal_quantile(p: float, mu: float | np.ndarray, sigma: float | np.ndarray):
    """calculates the quantile function for a normal distribution with mean mu,
    standard deviation sigma, for a given value p.

    Parameters
    ----------
    p : float
        value at which teh quantile function should be evaluated
    mu : float or np.ndarray
        mean
    sigma : float or np.ndarray
        standard deviation

    Returns
    -------
    float or np.ndarray
        quantile(s)
    """

    return norm.ppf(q=p, loc=mu, scale=sigma)

def get_upper_lower_bounds(p: float, mu: np.ndarray, sigma: np.ndarray):
    """calculate the upper and lower bounds of a prediction interval for a normal
    distribution with mean mu, standard deviation sigma, for a given value p.

    Parameters
    ----------
    p : float
        p value of the quantile function. For a 95% prediction interval, p=0.975
    mu : np.ndarray
        mean vector of the normal distribution
    sigma : np.ndarray
        standard deviation vector of the normal distribution

    Returns
    -------
    np.ndarray, np.ndarray
        lower and upper bounds of the prediction interval
    """

    quantiles = normal_quantile(p=p, mu=mu, sigma=sigma)
    lower_bound = 2 * mu - quantiles
    upper_bound = quantiles

    return lower_bound, upper_bound


if __name__ == "__main__":
    p = 0.975
    mu = np.array([0, 0.5, .5])
    sigma = np.array([2.0,2.0,3.0])
    quantile = normal_quantile(p=p, mu=mu, sigma=sigma)
    print(quantile)
    lower, upper = get_upper_lower_bounds(0.975, 6, 2)
    print(lower, upper)




