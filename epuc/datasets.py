import numpy as np
import torch

from torch.utils.data import Dataset
from epuc.configs import data_config


def sine_fct_prediction(x, freq: float = 10.0):
    pred_fct = lambda x: 0.5 * np.sin(freq * x) + 0.5
    return pred_fct(x)


def generate_bernoulli_labels(x_inst: np.ndarray, fct_pred):
    preds = fct_pred(x_inst)

    # sample labels from multinomial distribution
    labels = np.random.binomial(1, preds)

    return labels, preds


class BernoulliSineDataset(Dataset):
    def __init__(
        self,
        n_samples_1: int,
        n_samples_2: int,
        sine_factor: int = 5,
        x_min: float = 1.0,
        x_max: float = 1.0,
        x_split: float = 0.5,
    ):
        """Dataset for binary classification, where the labels follow a Bernoulli distribution whose
        parameter follows a sine function.

        Parameters
        ----------
        n_samples_1 : int
            number of samples in the interval [x_min, x_split]
        n_samples_2 : int
            number of samples in the interval [x_split, x_max]
        sine_factor : int, optional
            frequency of the sine function, by default 5
        x_min : float, optional
            lower bvound of the instance values, by default 1.0
        x_max : float, optional
            upper bound of the instance values, by default 1.0
        x_split : float, optional
            split value of the two intervals in which instances lie, by default 0.5
        """
        super().__init__()
        self.n_samples_1 = n_samples_1
        self.n_samples_2 = n_samples_2
        self.sine_factor = sine_factor
        self.split = x_split
        # instances within range
        self.x_inst_in = torch.from_numpy(
            np.random.uniform(x_min, x_split, n_samples_1)
        ).float()
        # instances outside range
        self.x_inst_out = torch.from_numpy(
            np.random.uniform(x_split, x_max, n_samples_2)
        ).float()
        if n_samples_2 > 0:
            self.x_inst = torch.cat((self.x_inst_in, self.x_inst_out), dim=0)
        else:
            self.x_inst = self.x_inst_in
        # generate labels whose class probability is given by the sine function
        fct_pred = lambda x: sine_fct_prediction(x, freq=self.sine_factor)
        self.y_labels, self.preds = generate_bernoulli_labels(self.x_inst, fct_pred)

    def __len__(self):
        return self.n_samples_1 + self.n_samples_2

    def __getitem__(self, index):
        return self.x_inst[index], self.y_labels[index]


class SineRegressionDataset(Dataset):
    def __init__(
        self,
        n_samples_1: int,
        n_samples_2: int = 0,
        sine_factor: int = 5,
        x_min: float = 0.0,
        x_max: float = 1.0,
        x_split: float = 0.5,
        eps_std: float = 0.01,
    ):
        """Dataset for regression, where the labels are given by
        a sine function with some added noise.

        Parameters
        ----------
        n_samples_1 : int
            number of samples in the interval [x_min, x_split]
        n_samples_2 : int, optional
            number of samples in the interval [x_split, x_max], by default 0
        sine_factor : int, optional
            frequency of the sine function, by default 5
        x_min : float, optional
            lower bound of the first interval for the instance values, by default 0.0
        x_max : float, optional
            upper bound for the instance values, by default 1.0
        x_split : float, optional
            split value of the two intervals in shich instances lie, by default 0.5
        eps_std : float, optional
            standard deviation of the added noise, by default 0.01
        """
        super().__init__()
        self.n_samples = n_samples_1 + n_samples_2
        self.sine_factor = sine_factor
        self.split = x_split
        # instances within range of first interval
        self.x_inst_in = torch.from_numpy(
            np.random.uniform(x_min, x_max, n_samples_1)
        ).float()
        # add small noise to the data (homoscedastic)
        eps = torch.normal(torch.zeros(n_samples_1 + n_samples_2), eps_std)
        if n_samples_2 > 0:
            self.x_inst_out = torch.from_numpy(
            np.random.uniform(x_max, 1, n_samples_2)
            ).float()
            self.x_inst = torch.cat((self.x_inst_in, self.x_inst_out), dim=0)
        else:
            self.x_inst = self.x_inst_in
        self.y_targets = sine_fct_prediction(self.x_inst, freq=self.sine_factor).float()
        self.y_targets += eps

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x_inst[index], self.y_targets[index]


if __name__ == "__main__":
    data_params = data_config["SineRegression"]
    dataset = SineRegressionDataset(**data_params)
