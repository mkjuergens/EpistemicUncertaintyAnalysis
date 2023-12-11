import torch
from torch import nn


class PredictorModel(nn.Module):
    """
    Simple neural network for predicting the probability of a binary event.
    """

    def __init__(self, hidden_dim: int = 10, output_dim: int = 1, use_relu: bool = False,
                 use_softplus: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.use_relu = use_relu
        self.softplus = nn.Softplus()
        self.use_softplus = use_softplus
        self.sigmoid = nn.Sigmoid() # binary setting: probability for class 1

    def forward(self, x):
        if self.use_relu:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
        elif self.use_softplus:
            x = self.softplus(self.fc1(x))
            x = self.softplus(self.fc2(x))
        else:
            x = self.fc1(x)
            x = self.fc2(x)
        x = self.sigmoid(self.fc3(x))

        return x

    
class RegressorModel(nn.Module):
    """
    Simple neural network for predicting the mean and variance of a normally distributed 
    random variable, in dependence of the input.
    """

    def __init__(self, hidden_dim: int = 10, use_softplus: bool = True, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()
        self.use_softplus = use_softplus

    def forward(self, x):
        if self.use_softplus:
            x = self.softplus(self.fc1(x))
            x = self.softplus(self.fc2(x))
        else:
            x = self.fc1(x)
            x = self.fc2(x)
        
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))

        return mu, sigma
    

class BetaNN(nn.Module):
    """
    Simple neural network for predicting the parameters of (uni- or multivariate) a Beta distribution.
    """

    def __init__(self,hidden_dim: int = 10, use_softplus: bool = True, output_dim: int = 1):
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

    def __init__(self, hidden_dim: int = 10, use_softplus: bool = True, output_dim: int = 1,):
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