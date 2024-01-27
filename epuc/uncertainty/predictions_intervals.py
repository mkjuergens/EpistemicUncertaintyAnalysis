import numpy as np
from scipy import stats

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

    return stats.norm.ppf(q=p, loc=mu, scale=sigma)

def get_upper_lower_bounds_normal(p: float, mu: np.ndarray, sigma: np.ndarray):
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


def get_upper_lower_bounds_empirical(p: float, y: np.ndarray):
    """calculate the (empirical) upper and lower bounds of a prediction interval, given the predictions of a set 
    of models.

    Parameters
    ----------
    p : float
        confidence level, must be in [0,1]
    y : np.ndarray of shape (n_samples, n_models)
        predictions of the models
    """

    lower_perc = (1 - p) / 2 * 100
    upper_perc = (1 + p) / 2 * 100

    # calculate lower and upper confidence bounds
    lower_bound = np.percentile(y, lower_perc, axis=1)
    upper_bound = np.percentile(y, upper_perc, axis=1)

    return lower_bound, upper_bound


def get_upper_lower_bounds_inv_gamma(p: float, alpha: np.ndarray, beta: np.ndarray):
    """calculate the upper and lower bounds of a prediction interval for a inverse gamma

    Parameters
    ----------
    p : float
        confidence level, must be in [0,1]
    alpha : np.ndarray of shape (n_instances, 1)
        alpha parameter of the inverse gamma distribution
    beta : np.ndarray
        beta parameter of the inverse gamma distribution

    Returns
    -------
    lower_bound, upper_bound : np.ndarray, np.ndarray
        confidence bounds per instance
    """

    tail_prob = (1 - p) / 2

    # calculate quantiles 
    lower_bound = stats.invgamma.ppf(tail_prob, alpha, scale=beta)
    upper_bound = stats.invgamma.ppf(1 - tail_prob, alpha, scale=beta)

    return lower_bound, upper_bound

def get_upper_lower_bounds_beta(p: float, alpha: np.ndarray, beta: np.ndarray):
    """calculate the upper and lower bounds of a prediction interval for a beta distribution

    Parameters
    ----------
    p : float
        confidence level, must be in [0,1]
    alpha : np.ndarray of shape (n_instances, 1)
        alpha parameter of the beta distribution
    beta :  np.ndarray of shape (n_instances, 1)
        beta parameter of the beta distribution

    Returns
    -------
    lower_bound, upper_bound : np.ndarray, np.ndarray
        confidence bounds per instance
    """

    tail_prob = (1 - p) / 2

    # calculate quantiles
    lower_bound = stats.beta.ppf(tail_prob, alpha, beta)
    upper_bound = stats.beta.ppf(1 - tail_prob, alpha, beta)

    return lower_bound, upper_bound

if __name__ == "__main__":
    p = 0.975
    mu = np.array([0, 0.5, .5])
    sigma = np.array([2.0,2.0,3.0])
    quantile = normal_quantile(p=p, mu=mu, sigma=sigma)
    print(quantile)
    lower, upper = get_upper_lower_bounds_normal(0.975, 6, 2)
    print(lower, upper)
    x = np.random.randn(1, 10)
    lower, upper = get_upper_lower_bounds_empirical(0.95, x)
    print(lower, upper)




