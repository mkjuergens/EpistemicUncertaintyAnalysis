import torch
import numpy as np

from scipy.stats import wasserstein_distance, beta
from statsmodels.distributions.empirical_distribution import ECDF


def ecdf_values_ensemble(ens_models: list, x_val: np.ndarray, theta_vals: np.ndarray):
    """calculates the empirical distribution function for a spepciofic value  
    approximated by the ensmeble model.

    Parameters
    ----------
    ens_models : list
        list of models which form the ensmeble
    x_val : np.ndarray
        value at which the ecd is calculated
    theta_vals: np.ndarray
        values of theta for which the ecdf is calculated

    Returns
    -------
    np.ndarray
        empirical distribution function evaluated at the defined theta values
    """

    # save the predictions of the ensemble models
    preds_ens = np.zeros(len(ens_models))
    for i, model in enumerate(ens_models):
        model.eval()
        with torch.no_grad():
            ens_pred = model(torch.tensor(x_val, dtype=torch.float32)).numpy()
            preds_ens[i] = ens_pred
    # calculate the empirical distribution function
    ecdf_ens = ECDF(preds_ens)
    # calculate values of ecdf for theta values
    ecdf_vals = np.zeros(len(theta_vals))
    for i, theta in enumerate(theta_vals):
        ecdf_vals[i] = ecdf_ens(theta)

    return ecdf_vals

def parametric_cdf_values_beta(model: torch.nn.Module, x_val: np.ndarray, theta_vals: np.ndarray):
    """calculates the cdf of the beta distribution for a spepciofic value  
    approximated by the ensmeble model.

    Parameters
    ----------
    model : torch.nn.Module
        model which forms the second order model
    x_val : np.ndarray
        value at which the ecd is calculated
    theta_vals: np.ndarray
        values of theta for which the ecdf is calculated

    Returns
    -------
    np.ndarray
        cdf of beta distribution evaluated at the defined theta values
    """
    # calculate the parameters of the beta distribution
    model.eval()
    with torch.no_grad():
        alpha_param, beta_param = model(torch.tensor(x_val, dtype=torch.float32))

    # beta distribution with predicted parameters
    beta_dist = beta(alpha_param.numpy(), beta_param.numpy())
    cdf_vals = np.zeros(len(theta_vals))
    for i, theta in enumerate(theta_vals):
        cdf_vals[i] = beta_dist.cdf(theta)

    return cdf_vals

def parametric_cdf_values_second_order(model: torch.nn.Module, x_val: np.ndarray,
                                        theta_vals: np.ndarray, distribution: str = "beta"):
    
    if distribution == "beta":
        return parametric_cdf_values_beta(model, x_val, theta_vals)
    else:
        raise ValueError("Distribution not implemented")


def compute_wasserstein_distance(
    ens_models: list, second_order_model: torch.nn.Module, x_values: np.ndarray,
    theta_interval: tuple = (0, 1), n_splits: int = 1000, distribution: str = "beta"):
    """Compute the Wasserstein distance between the ensemble members and the second order model.

    Parameters
    ----------
    ens_models : list
        list of ensemble models
    second_order_model : torch.nn.Module
        second order model
    data : np.ndarrays
        data to compute the distance on

    Returns
    -------
    list
        list of empirical Wasserstein distances evaluated per instance value
    """
    distances = np.zeros(len(x_values))
    for i in range(len(x_values)):
        x_val = x_values[i].reshape(-1,1)
        # discretize theta space to n_splits
        theta_vals = np.linspace(theta_interval[0], theta_interval[1], n_splits)
        # caluculate empirical distribution function evaluated at theta of ensemble model
        ecdf_values_ens = ecdf_values_ensemble(ens_models, x_val, theta_vals)
        # caculate cdf of beta distribution evaliuated at theta of second order model 
        cdf_values_second_order = parametric_cdf_values_second_order(
            second_order_model, x_val, theta_vals, distribution
        )
        # split interval of possible target values into n_splits
        distances[i] = wasserstein_distance(ecdf_values_ens, cdf_values_second_order)

    return distances