import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from epuc.model import PredictorModel, RegressorModel, BetaNN


def train_model(model, dataloader, criterion, n_epochs: int, optim, **kwargs):
    """helper function to train a specific model using a specific criterion and optimizer.

    Parameters
    ----------
    model : _type_
        torch model taking as input the instances from the dataloader batch
    dataloader : torch.utils.data.DataLoader
        dataloader containing samples and labels
    criterion : loss criterion
        loss whihc takes as input the output of the model plus the labels
    n_epochs : int
        number of epochs to train the model
    optim : torch.optim optimizer
        optimizer to use for training

    Returns
    -------
    torch model
        trained model
    """

    model.train()
    for epoch in range(n_epochs):
        for x, y in dataloader:
            optim.zero_grad()
            x = x.view(-1, 1)
            y = y.view(-1, 1).float()  # use float for BCE loss
            pred = model(x)
            loss = criterion(pred, y, **kwargs)
            loss.backward()
            optim.step()

    return model


def train_multiple_models_primary_loss_beta(
    dataset,
    n_samples: int,
    loss,
    batch_size: int,
    lr: float,
    n_epochs: int,
    n_runs: int,
    n_samples_2=0,
    **kwargs
):
    """train multiple models, save predictions and instance values for each run.

    Parameters
    ----------
    dataset : _type_
        dataset containing instacnes and labels
    n_samples : int
        number of samples to generate for the dataset
    loss : _type_
        _description_
    batch_size : int
        _description_
    lr : float
        _description_
    n_epochs : int
        _description_
    n_runs : int
        _description_

    Returns
    -------
    _type_
        _description_
    """

    results = np.zeros(
        (n_runs, n_samples + n_samples_2)
    )  # save results (i.e. predictions) for multiple runs here
    x_inst = np.zeros(
        (n_runs, n_samples + n_samples_2)
    )  # save instance values for each run here
    for run in range(n_runs):
        model = PredictorModel(use_relu=False, use_softplus=True)
        if n_samples_2 > 0:
            dataset_train = dataset(n_samples_1=n_samples, n_samples_2=n_samples_2)
        else:
            dataset_train = dataset(n_samples=n_samples)
        loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        model_out = train_model(model, loader, loss, n_epochs, optim, **kwargs)
        # sort instances

        pred_probs = (
            model_out(torch.sort(dataset_train.x_inst)[0].view(-1, 1))
            .detach()
            .numpy()
            .squeeze()
        )
        results[run, :] = pred_probs
        x_inst[run, :] = torch.sort(dataset_train.x_inst)[0].detach().numpy().squeeze()

    return results, x_inst


def sample_predictions_secondary_beta(
    alphas: torch.tensor, betas: torch.tensor, n_samplings: int, **kwargs
):
    assert alphas.shape == betas.shape, "alphas and betas must have the same shape"
    results = np.zeros(
        (n_samplings, alphas.shape[0])
    )  # save results (i.e. sampled predictions) for multiple samplings here
    for sampling in range(n_samplings):
        # sample predictive probabilities fro class 1 from induced Beta distribution
        theta = np.random.beta(
            alphas.detach().numpy(), betas.detach().numpy()
        ).squeeze()
        results[sampling, :] = theta

    return results


def train_muliple_models_primary_nig(
    dataset,
    n_samples,
    n_samples_2,
    x_max: float,
    loss,
    batch_size: int,
    eps_var: float,
    lr: float,
    n_epochs: int,
    n_runs: int,
):
    results = np.zeros((n_runs, n_samples + n_samples_2))
    x_inst = np.zeros((n_runs, n_samples + n_samples_2))  # save instance values here
    for run in range(n_runs):
        model = RegressorModel()
        dataset_train = dataset(
            n_samples=n_samples, x_max=x_max, n_samples_2=n_samples_2, eps_var=eps_var
        )
        loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        model_out = train_model(
            model,
            dataloader=loader,
            criterion=loss,
            n_epochs=n_epochs,
            optim=optim
        )

        # output parameters of normal distribution
        mu, sigma = model_out(torch.sort(dataset_train.x_inst)[0].view(-1, 1))
        # sample from the induced distribution
        y_targets = torch.normal(mean=mu.detach(), std=sigma.detach()).detach().numpy().squeeze()
        results[run, :] = y_targets
        x_inst[run, :] = torch.sort(dataset_train.x_inst)[0].detach().numpy().squeeze()

    return results, x_inst

