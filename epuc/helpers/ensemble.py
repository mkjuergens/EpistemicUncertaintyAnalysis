import torch
from epuc.helpers.train import train_model
from epuc.models import create_model
from epuc.datasets import SineRegressionDataset


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
        self.losses = []
    def train(
        self,
        dataset,
        data_params,
        train_params: dict,
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
        """
        for model in self.models:
            dataset_train = dataset(**data_params)
            data_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size=train_params["batch_size"], shuffle=True
            )
            model, loss = train_model(
                model,
                data_loader,
                criterion=train_params["loss"],
                n_epochs=train_params["n_epochs"],
                optim=train_params["optim"],
                return_loss=True,
                **train_params["optim_kwargs"]
            )
            self.losses.append(loss)

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
        preds = [model(x) for model in self.models]
        if isinstance(preds[0], torch.Tensor):
            preds_out = torch.stack(preds)
        elif isinstance(preds[0], (list, tuple)):
            preds_out = torch.zeros(x.shape[0], len(preds), len(preds[0]))
            for pred in range(len(preds[0])):
                for ens in range(len(preds)):
                    preds_out[:, ens, pred] = preds[ens][pred].squeeze()
        else:
            raise TypeError("unsupported prediction type {}".format(type(preds[0])))
        return preds_out

    def predict_mean(self, x):
        preds = self.predict(x)
        # predict mean over ensemble members
        return preds.mean(dim=1)
    
class GaussianEnsemble(Ensemble):

    def __init__(self, model_config: dict, ensemble_size: int):
        super().__init__(model_config, ensemble_size)

    def predict_mean_var(self, x):

        preds = self.predict(x) # of shape (n_instances, n_ensemble, 2)
        pred_mean = preds[:, :, 0].mean(dim=1)
        # use formula for variance of gauusian mixture model
        preds_var = (preds[:, :, 1]**2 + preds[:, :, 0] ** 2).mean(dim=1) - pred_mean ** 2
        preds_std = torch.sqrt(preds_var)
        return pred_mean, preds_std


if __name__ == "__main__":
    from epuc.configs import model_config, data_config, train_config
    ensemble = Ensemble(model_config["Normal"], ensemble_size=2)
    data_params = data_config["SineRegression"]
    train_params = train_config["Normal"]
    ensemble.train(SineRegressionDataset, data_params, train_params)
    x = torch.linspace(0, 1, 100).view(-1, 1)
    preds = ensemble.predict_mean(x)