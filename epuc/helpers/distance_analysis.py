import torch
import numpy as np

from scipy.stats import wasserstein_distance, beta, norm, invgamma
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

def parametric_cdf_values_NIG(model: torch.nn.Module, x_val: np.ndarray, mu_vals: np.ndarray,
                              sigma_vals: np.ndarray):
    model.eval()
    with torch.no_grad():
        # sample parameters of NIG distribution from model
        gamma, nu, alpha, beta = model(torch.tensor(x_val, dtype=torch.float32))

    # mu is normally distributed with parameters gamma, beta/(alpha -1)*nu
    mu_dist = norm(loc=gamma.numpy(), scale=np.sqrt((beta/(alpha -1)*nu).numpy()))
    digma_dist = invgamma()


def parametric_cdf_values_second_order(model: torch.nn.Module, x_val: np.ndarray,
                                        theta_vals: np.ndarray, distribution: str = "beta"):
    
    if distribution == "beta":
        return parametric_cdf_values_beta(model, x_val, theta_vals)
    else:
        raise ValueError("Distribution not implemented")


def compute_wasserstein_distance(
    ens_models: list, second_order_models: list | torch.nn.Module, x_values: np.ndarray,
    theta_interval: tuple = (0, 1), n_splits: int = 1000, distribution: str = "beta"):
    """Compute the Wasserstein distance between the ensemble members and the second order model.

    Parameters
    ----------
    ens_models : list
        list of ensemble models
    second_order_models : list or torch.nn.Module
        second order model(s). If multiple models are given as input, the avergae of
        the distance to the first order models ecdf is computed. 
    data : np.ndarrays
        data to compute the distance on

    Returns
    -------
    list
        list of empirical Wasserstein distances evaluated per instance value
    """
    distances = np.zeros(len(x_values))
    if isinstance(second_order_models, list):
        #save variance of the distances
        var_dists = np.zeros(len(x_values))
    for i in range(len(x_values)):
        x_val = x_values[i].reshape(-1,1)
        # discretize theta space to n_splits
        theta_vals = np.linspace(theta_interval[0], theta_interval[1], n_splits)
        # caluculate empirical distribution function evaluated at theta of ensemble model
        ecdf_values_ens = ecdf_values_ensemble(ens_models, x_val, theta_vals)
        # caculate cdf of beta distribution evaliuated at theta of second order model 
        # check if multiple second order models are given
        if isinstance(second_order_models, list):
            # calculate the average distance to the ensemble models ecdf
            # also save variance of the distances
            dist_second_order = []
            for model in second_order_models:
                cdf_values_second_order = parametric_cdf_values_second_order(
                    model, x_val, theta_vals, distribution
                )
                dist_second_order.append(wasserstein_distance(ecdf_values_ens, cdf_values_second_order))
            dist_second_order = np.mean(dist_second_order)
            var_dist = np.var(dist_second_order)
            var_dists[i] = var_dist
        else:
            cdf_values_second_order = parametric_cdf_values_second_order(
                second_order_models, x_val, theta_vals, distribution
            )
            dist_second_order = wasserstein_distance(ecdf_values_ens, cdf_values_second_order)
        distances[i] = dist_second_order


    return distances if not isinstance(second_order_models, list) else distances, var_dists