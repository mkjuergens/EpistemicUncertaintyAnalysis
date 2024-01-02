import torch
from torch import nn


class PredictorModel(nn.Module):
    """
    Simple neural network for predicting the probability of a binary event.
    """

    def __init__(
        self,
        hidden_dim: int = 10,
        output_dim: int = 1,
        n_hidden_layers: int = 1,
        use_softplus: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.layers = [nn.Linear(1, hidden_dim), nn.Softplus() if use_softplus else nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Softplus() if use_softplus else nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


class RegressorModel(nn.Module):
    """
    Simple neural network for predicting the mean and variance of a normally distributed
    random variable, in dependence of the input.
    """

    def __init__(
        self, hidden_dim: int = 10, use_softplus: bool = True, n_hidden_layers: int = 1,
         output_dim: int = 1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        self.layers = [nn.Linear(1, hidden_dim), nn.Softplus() if use_softplus else nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Softplus() if use_softplus else nn.ReLU())

        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)

        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        
        x = self.model(x)

        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))

        return mu, sigma


class BetaNN(nn.Module):
    """
    Simple neural network for predicting the parameters of (uni- or multivariate) a Beta distribution.
    """

    def __init__(
        self, hidden_dim: int = 10, use_softplus: bool = True, output_dim: int = 1
    ):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_alpha = nn.Linear(hidden_dim, output_dim)
        self.fc_beta = nn.Linear(hidden_dim, output_dim)
        self.use_softplus = use_softplus
        self.softplus = nn.Softplus()

    def forward(self, x):
        if self.use_softplus:
            x = self.softplus(self.fc1(x))
            x = self.softplus(self.fc2(x))
        else:
            x = self.fc1(x)
            x = self.fc2(x)

        alpha = torch.exp(self.fc_alpha(x))
        beta = torch.exp(self.fc_beta(x))

        return alpha, beta


class NIGNN(nn.Module):
    """
    Simple neural network for predicting the parameters of a (uni-or multivariate)
    normal inverse gamma distribution.
    """

    def __init__(
        self,
        hidden_dim: int = 10,
        use_softplus: bool = True,
        output_dim: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_1 = nn.Linear(1, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)

        # parameters of the normal inverse gamma distribution
        self.fc_gamma = nn.Linear(hidden_dim, 1)
        self.fc_nu = nn.Linear(hidden_dim, 1)
        self.fc_alpha = nn.Linear(hidden_dim, 1)
        self.fc_beta = nn.Linear(hidden_dim, 1)

        self.use_softplus = use_softplus
        self.softplus = nn.Softplus()

    def forward(self, x):
        if self.use_softplus:
            x = self.softplus(self.fc_1(x))
            x = self.softplus(self.fc_2(x))
        else:
            x = self.fc_1(x)
            x = self.fc_2(x)

        gamma = self.fc_gamma(x)
        nu = self.softplus(self.fc_nu(x))
        alpha = self.softplus(self.fc_alpha(x)) + 1
        beta = self.softplus(self.fc_beta(x))

        return gamma, nu, alpha, beta


def create_model(config):
    """
    Create a model from a configuration dictionary.
    """
    model = config["model"](**config["kwargs"])
    return model


if __name__ == "__main__":
    from epuc.configs import model_config

    model = create_model(model_config["Normal"])
    print(model)
    x = torch.randn(10, 1)
    print(model(x))
