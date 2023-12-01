import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from epuc.model import PredictorModel, RegressorModel, BetaNN, NIGNN, create_model


# TODO: generalize functions for different models, losses, datasets, etc.



def train_model(
    model,
    dataloader,
    criterion,
    n_epochs: int,
    optim,
    return_loss: bool = False,
    **kwargs
):
    """
    helper function to train a specific model using a specific criterion and optimizer.

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
    return_loss : bool, optional
        whether to return the loss, by default False

    Returns
    -------
    torch model
        trained model
    """

    model.train()
    optimizer = optim(model.parameters(), **kwargs)
    if return_loss:
        loss_list = []
    for epoch in range(n_epochs):
        loss_epoch = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            x = x.view(-1, 1)
            y = y.view(-1, 1).float()  # use float for BCE loss
            pred = model(x)
            loss = criterion(pred, y)
            if return_loss:
                loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
        if return_loss:
            loss_list.append(loss_epoch / len(dataloader))

    return model if not return_loss else (model, loss_list)


def train_multiple_models_secondary_loss_beta(
    dataset,
    n_samples: int,
    loss,
    batch_size: int,
    lr: float,
    n_epochs: int,
    n_runs: int,
    n_samples_2: int = 0,
    **kwargs
):
    """train multiple models for predicting the parameters of the beta distribution,
    save predictions and instance values for each run.
    """
    results = np.zeros(
        (n_runs, n_samples + n_samples_2, n_samples + n_samples_2)
    )  # save results (i.e. parameters of alpha and beta) for multiple runs here
    x_inst = np.zeros(
        (n_runs, n_samples + n_samples_2)
    )  # save instance values for each run here
    for run in range(n_runs):
        model = BetaNN()
        if n_samples_2 > 0:
            dataset_train = dataset(n_samples_1=n_samples, n_samples_2=n_samples_2)
        else:
            dataset_train = dataset(n_samples=n_samples)
        loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        model_out = train_model(model, loader, loss, n_epochs, optim, **kwargs)
        # sort instances

        alpha, beta = (
            model_out(torch.sort(dataset_train.x_inst)[0].view(-1, 1))
        )
        results[run, :, 0] = alpha.detach().numpy().squeeze()
        results[run, :, 1] = beta.detach().numpy().squeeze()
        x_inst[run, :] = torch.sort(dataset_train.x_inst)[0].detach().numpy().squeeze()

    return results, x_inst


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
        dataset containing instances and labels
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
    # detach and covnert to numpy if alpha, beta are tensors
    alphas = alphas.detach().numpy() if isinstance(alphas, torch.Tensor) else alphas
    betas = betas.detach().numpy() if isinstance(betas, torch.Tensor) else betas
    for sampling in range(n_samplings):
        # sample predictive probabilities fro class 1 from induced Beta distribution
        theta = np.random.beta(
            alphas, betas
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
            model, dataloader=loader, criterion=loss, n_epochs=n_epochs, optim=optim
        )

        # output parameters of normal distribution
        mu, sigma = model_out(torch.sort(dataset_train.x_inst)[0].view(-1, 1))
        # sample from the induced distribution
        y_targets = (
            torch.normal(mean=mu.detach(), std=sigma.detach())
            .detach()
            .numpy()
            .squeeze()
        )
        results[run, :] = y_targets
        x_inst[run, :] = torch.sort(dataset_train.x_inst)[0].detach().numpy().squeeze()

    return results, x_inst
