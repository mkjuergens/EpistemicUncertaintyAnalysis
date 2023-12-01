import torch


from epuc.helpers.train import train_model
from epuc.model import PredictorModel, RegressorModel, BetaNN, NIGNN, create_model
from epuc.configs import model_config, train_config
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

    def train(
        self,
        dataset,
        n_samples_1,
        n_samples_2,
        x_split: float,
        eps_var: float,
        train_params: dict,
    ):
        for model in self.models:
            dataset_train = dataset(
                n_samples_1=n_samples_1,
                n_samples_2=n_samples_2,
                x_max=x_split,
                eps_var=eps_var)
            data_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size=train_params["batch_size"], shuffle=True
            )
            train_model(
                model,
                data_loader,
                criterion=train_params["loss"],
                n_epochs=train_params["n_epochs"],
                optim=train_params["optim"],
                return_loss=False,
                **train_params["optim_kwargs"]
            )

if __name__ == "__main__":
    ensemble = Ensemble(model_config["Normal"], ensemble_size=10)
    ensemble.train(
        dataset=SineRegressionDataset,
        n_samples_1=100,
        n_samples_2=100,
        x_split=0.5,
        eps_var=0.01,
        train_params=train_config["Normal"],
    )

    print(ensemble.models[0].fc1.weight)