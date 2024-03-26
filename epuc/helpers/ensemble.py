import os
from collections import defaultdict
from typing import Optional

import torch
from tqdm import tqdm
from epuc.helpers.train import train_model
from epuc.models import create_model
from epuc.datasets import SineRegressionDataset, PolynomialDataset


class Ensemble:
    def __init__(self, model_config: dict, ensemble_size: int):
        """
        Ensemble class for training multiple models and storing them in a list.

        Parameters
        ----------
        model_config : dict
            dictionary containing the configuration for the model
        ensemble_size : int
            number of models to train
        """
        self.model_config = model_config
        self.ensemble_size = ensemble_size
        self.models = [create_model(model_config) for _ in range(ensemble_size)]
        self.dict_mean_params = {}
        self.dict_std_params = {}
        self.dict_losses = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_config["model"].__name__ # for saving models

    def train(
        self,
        dataset,
        data_params,
        train_params: dict,
        return_mean_params: bool = False,
        return_std_params: bool = False,
        resample_train_data: bool = True,
        x_eval: Optional[torch.tensor] = None,
    ):
        """train an ensemble of models based on the same (resampled) dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            has to include the options n_samples_1, n_samples_2, x_max
        data_params : dict
            dictionary containing the parameters for the dataset
        train_params : dict
            dictionary containing the parameters for training
        return_mean_params : bool, optional
            whether to return the mean parameters of the ensemble (per epoch), by default False
        return_std_params : bool, optional
            whether to return the standard deviation of the ensemble parameters (per epoch),
            by default False
        resample_train_data : bool, optional
            whether to resample the training data for each ensemble member, by default True
        x_eval : torch.tensor, optional
            tensor of instances to evaluate the model on, by default None
        """
        print(f"Training {self.ensemble_size} models on {self.device}.")
        if return_mean_params:
            # dictionary for each ensemble member
            ensemble_dicts = [defaultdict(list) for _ in range(self.ensemble_size)]

        dataset_train = dataset(**data_params)
        for i, model in enumerate(tqdm(self.models)):
            if resample_train_data:
                # resample the training data for each ensemble member
                dataset_train = dataset(**data_params)
            data_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size=train_params["batch_size"], shuffle=True
            )
            model, dict_returns = train_model(
                model,
                data_loader,
                criterion=train_params["loss"],
                n_epochs=train_params["n_epochs"],
                optim=train_params["optim"],
                return_loss=True,
                return_params=return_mean_params, 
                x_eval=x_eval,
                device=self.device,
                **train_params["optim_kwargs"],
            )
            self.dict_losses[i] = dict_returns["loss"]
            if return_mean_params:
                ensemble_dicts[i] = {
                    k: dict_returns[k] for k in dict_returns.keys() if k != "loss"
                }
        # now average the list of dictionaries in ensemble_dicts to get one list per key    
        if return_mean_params:
            for k in ensemble_dicts[0].keys():
                self.dict_mean_params[k] = torch.stack(
                    [torch.tensor(d[k]) for d in ensemble_dicts], axis=0
                )# TODO: check what it does here?

        if return_std_params:
            # ersturn standard deviation betweeen meab predictions of ensemble members
            for k in ensemble_dicts[0].keys():
                self.dict_std_params[k] = torch.stack(
                    [torch.tensor(d[k]) for d in ensemble_dicts], axis=0
                ).std(axis=0)


    def predict(self, x):
        """
        Predict the output of the ensemble for a given input x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            tensor containing the ensemble predictions
        """
        # put tensor on device
        x = x.to(self.device)
        preds = [model(x) for model in self.models]
        if isinstance(preds[0], torch.Tensor):
            # back on cpu if on gpu
            preds_out = torch.stack(preds, axis=1).cpu()
        elif isinstance(preds[0], (list, tuple)):
            preds_out = torch.zeros(x.shape[0], len(preds), len(preds[0]))
            for pred in range(len(preds[0])):
                for ens in range(len(preds)):
                    preds_out[:, ens, pred] = preds[ens][pred].squeeze().cpu()
        else:
            raise TypeError("unsupported prediction type {}".format(type(preds[0])))
        return preds_out

    def predict_mean(self, x):
        preds = self.predict(x)
        # predict mean over ensemble members (axis 1)
        return preds.mean(dim=1)

    def predict_std(self, x):
        preds = self.predict(x)
        # predict mean over ensemble members
        return preds.std(dim=1)

    # add method to save models
    def save_models(self, path):
        # create directory if it does not exist
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path + f"/model_{i}.pkl")
            # save loss for each model
            torch.save(self.dict_losses[i], path + f"/loss_{i}.pkl")

    def load_models(self, path):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(path + f"/model_{i}.pkl"))
            self.dict_losses[i] = torch.load(path + f"/loss_{i}.pkl")


class GaussianEnsemble(Ensemble):
    """Ensmeble of models which predict a Gaussian distribution, that is, the mean and standard
    deviation for each instance assuming the target values follow a Gaussian distribution.
    """

    def __init__(self, model_config: dict, ensemble_size: int):
        """
        Parameters
        ----------
        model_config : dict
            dictionary of model parameters
        ensemble_size : int
            number of ensemble members
        """
        super().__init__(model_config, ensemble_size)

    def predict_mean_params(self, x):
        """
        Parameters
        ----------
        x : torch.tensor
            tensor of instances

        Returns
        -------
        torch.tensor, torch.tensor
            mean and variance of the ensemble
        """

        preds = self.predict(x)  # of shape (n_instances, n_ensemble, 2)
        mean_mu = preds[:, :, 0].mean(dim=1)
        # use formula for variance of gauusian mixture model
        # mean_var = (preds[:, :, 1] ** 2 + preds[:, :, 0] ** 2).mean(
        #    dim=1
        # ) - pred_mean**2
        mean_std = preds[:, :, 1].mean(dim=1)
        # preds_std = torch.sqrt(preds_var)
        return mean_mu, mean_std

    def predict_std_params(self, x):
        preds = self.predict(x)
        std_mu = preds[:, :, 0].std(dim=1)
        std_std = preds[:, :, 1].std(dim=1)

        return std_mu, std_std


class NIGEnsemble(Ensemble):
    """Ensemble of models predicting the parameters of a Normal-Inverse-Gamma distribution."""

    def __init__(self, model_config: dict, ensemble_size: int):
        super().__init__(model_config, ensemble_size)

    def predict_normal_params(self, x):
        """predict the mean of mu and sigma^2 (of the modeled normal distribution)
        as well as their respective variance.
        Here, we take the mean of the ensemble predictions as the mean of the NIG parameters.

        The formulas for the mean and varaince of the NIG distribution are used.

        Parameters
        ----------
        x : torch.tensor
            tensor of instances

        Returns
        -------
        mean_mu, mean_sigma2, var_mu, var_sigma2
        """

        # predict mean over ensemble members
        preds = self.predict_mean(x)
        gamma, nu, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        mean_mu = gamma
        var_mu = beta / (nu * (alpha - 1))
        mean_sigma2 = beta / (alpha - 1)  # mean of inverse gamma distribution
        var_sigma2 = beta**2 / (
            (alpha - 1) ** 2 * (alpha - 2)
        )  # variance of inverse gamma dist

        return mean_mu, var_mu, mean_sigma2, var_sigma2

    def predict_uc(self, x):
        preds = self.predict_mean(x)
        gamma, nu, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        # epistemic uncertainty: Var[mu]
        ep_uc = beta / (nu * (alpha - 1))
        # aleatoric uncertainty: E[sigma^2]
        al_uc = ep_uc * nu

        return ep_uc, al_uc


class BetaEnsemble(Ensemble):
    def __init__(self, model_config: dict, ensemble_size: int):
        super().__init__(model_config, ensemble_size)

    def predict_mean_p(self, x):
        preds = self.predict(x)
        pred_means = preds[:, :, 0] / (preds[:, :, 0] + preds[:, :, 1])
        return pred_means # TODO: CHECK: do not take mean over predictions, this fucks things up


if __name__ == "__main__":
    from epuc.configs import create_train_config, create_model_config, create_data_config
    ensemble = Ensemble(model_config=create_model_config()["Normal"], ensemble_size=2)

    data_params = create_data_config()["regression"]["polynomial"]
    train_params = create_train_config()["Normal"]
    ensemble.train(PolynomialDataset, data_params, train_params)
    x = torch.linspace(0, 1, 100).view(-1, 1)
    preds = ensemble.predict_mean(x)
    ensemble.save_models(path="./results/")
