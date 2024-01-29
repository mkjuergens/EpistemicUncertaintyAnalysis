from typing import Optional
import numpy as np
import torch

from torch.utils.data import Dataset


def sine_fct_prediction(x, freq: float = 1.0, amplitude: float = 0.8):
    pred_fct = lambda x: 0.5 * (amplitude * np.sin(2 * np.pi * freq * x) + 1)
    return pred_fct(x)


def polynomial_fct(x, degree: int = 4):
    pred_fct = lambda x: x**degree
    return pred_fct(x)


def linear_fct(x, slope: float = 0.5, intercept: float = 0.25):
    pred_fct = lambda x: slope * x + intercept
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
        n_samples_2: int = 0,
        sine_factor: int = 1.0,
        amplitude: float = 0.8,
        x_min: float = 0.0,
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
            lower bound of the instance values, by default 1.0
        x_max : float, optional
            upper bound of the instance values, by default 1.0
        x_split : float, optional
            split value of the two intervals in which instances lie, by default 0.5
        """
        super().__init__()
        self.n_samples_1 = n_samples_1
        self.n_samples_2 = n_samples_2
        self.sine_factor = sine_factor
        self.amplitude = amplitude
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
        fct_pred = lambda x: sine_fct_prediction(
            x, freq=self.sine_factor, amplitude=self.amplitude
        )
        self.y_labels, self.preds = generate_bernoulli_labels(self.x_inst, fct_pred)

    def __len__(self):
        return self.n_samples_1 + self.n_samples_2

    def __getitem__(self, index):
        return self.x_inst[index], self.y_labels[index]


class BernoulliLinearDataset(Dataset):
    def __init__(
        self,
        n_samples_1: int,
        n_samples_2: int = 0,
        x_min: float = 0.0,
        x_max: float = 1.0,
        slope: float = 0.5,
        intercept: float = 0.25,
        x_split: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples_1 + n_samples_2
        self.split = x_split
        # sample instances from uniform distribution
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

        fct_pred = lambda x: linear_fct(x, slope=slope, intercept=intercept)
        self.y_labels, self.preds = generate_bernoulli_labels(self.x_inst, fct_pred)

    def __len__(self):
        return self.n_samples

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
        self.eps_std = eps_std
        # instances within range of first interval
        self.x_inst_in = torch.from_numpy(
            np.random.uniform(x_min, x_split, n_samples_1)
        ).float()
        # add small noise to the data (homoscedastic)
        eps = torch.normal(torch.zeros(n_samples_1 + n_samples_2), eps_std)
        if n_samples_2 > 0:
            self.x_inst_out = torch.from_numpy(
                np.random.uniform(x_split, x_max, n_samples_2)
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


class PolynomialDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        degree: int = 3,
        x_min: int = -4,
        x_max: int = 4,
        eps_std: float = 3.0,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.degree = degree
        self.x_min = x_min
        self.x_max = x_max
        self.eps_std = eps_std

        self.x_inst = torch.from_numpy(
            np.random.uniform(x_min, x_max, n_samples)
        ).float()
        self.y_targets = polynomial_fct(self.x_inst, degree=self.degree).float()
        eps = torch.normal(torch.zeros(n_samples), eps_std)
        self.y_targets += eps

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x_inst[index], self.y_targets[index]


def create_evaluation_data(
    data_config,
    problem_type: str = "regression",
    data_type: str = "polynomial",
    n_eval_points: int = 1000,
    range_x_eval: Optional[tuple] = None,
):
    if problem_type == "regression":
        if data_type == "polynomial":
            dataset = PolynomialDataset
            dataset_eval = dataset(**data_config["regression"][data_type])
            if range_x_eval is not None:
                x_eval = torch.from_numpy(
                    np.linspace(range_x_eval[0], range_x_eval[1], n_eval_points)
                ).float()
            else:
                x_eval = torch.from_numpy(np.linspace(-6, 6, n_eval_points)).float()
            y_eval = polynomial_fct(
                x_eval, degree=data_config["regression"][data_type]["degree"]
            )
            x_train = dataset_eval.x_inst
            y_targets = dataset_eval.y_targets

        elif data_type == "sine":
            dataset = SineRegressionDataset
            dataset_eval = dataset(**data_config["regression"][data_type])
            x_eval = torch.from_numpy(np.linspace(0, 1, n_eval_points)).float()
            y_eval = sine_fct_prediction(
                x_eval, freq=data_config["regression"][data_type]["sine_factor"]
            )
            x_train = dataset_eval.x_inst
            y_targets = dataset_eval.y_targets

        else:
            raise NotImplementedError("data_type not implemented for regression")

    elif problem_type == "classification":
        if data_type == "sine":
            dataset = BernoulliSineDataset
            dataset_eval = dataset(**data_config["classification"][data_type])
            if range_x_eval is not None:
                x_eval = torch.from_numpy(
                    np.linspace(range_x_eval[0], range_x_eval[1], n_eval_points)
                ).float()
            else:
                x_eval = torch.from_numpy(np.linspace(0, 1, n_eval_points)).float()
            x_eval = torch.from_numpy(np.linspace(0,  1, n_eval_points)).float()
            y_eval = sine_fct_prediction(
                x_eval, freq=data_config["classification"][data_type]["sine_factor"]
            )
            x_train = dataset_eval.x_inst
            y_targets = dataset_eval.y_labels

        elif data_type == "linear":
            dataset = BernoulliLinearDataset
            dataset_eval = dataset(**data_config["classification"][data_type])
            x_eval = torch.from_numpy(np.linspace(0, 1, n_eval_points)).float()
            y_eval = linear_fct(
                x_eval,
                slope=data_config["classification"][data_type]["slope"],
                intercept=data_config["classification"][data_type]["intercept"],
            )
            x_train = dataset_eval.x_inst
            y_targets = dataset_eval.y_labels
        else:
            raise NotImplementedError("data_type not implemented for classification")

    else:
        raise NotImplementedError

    return dataset, x_eval, y_eval, x_train, y_targets


if __name__ == "__main__":
    dataset, x_eval, y_eval, x_train, y_targets = create_evaluation_data(
        problem_type="regression", data_type="polynomial"
    )
    print(dataset)
