import numpy as nn

from epuc.uncertainty import get_upper_lower_bounds_empirical


def compute_upper_lower_bounds_ensemble(results_dict: dict):

    dict_returns = {}
    for key in results_dict.keys():
        mean_params = results_dict[key].mean(dim=0)
        lower, upper = get_upper_lower_bounds_empirical(
            p=0.975, y=results_dict[key].T
        )  # transpose because we take confidence bounds over members
        # print(lower.shape, upper.shape)
        dict_returns[key] = {"lower": lower, "upper": upper, "mean": mean_params}

    return dict_returns
