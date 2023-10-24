import numpy as np
import torch
from torch import nn

class PredictorModel(nn.Module):
    """
    Simple neural network for predicting the probability of a binary event.
    """

    def __init__(self, n_hidden_dims: int = 10, out_dim: int = 1, use_relu: bool = False,
                 use_softplus: bool = True, *args, **kwargs) -> None:
        super().__init__()
        self.n_hidden_dims = n_hidden_dims

        self.fc1 = nn.Linear(1, n_hidden_dims)
        self.fc2 = nn.Linear(n_hidden_dims, n_hidden_dims)
        self.fc3 = nn.Linear(n_hidden_dims, out_dim)
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
    

class BetaNN(nn.Module):
    """
    Simple neural network for predicting the parameters of a Beta distribution.
    """

    def __init__(self,n_hidden: int = 10, use_softplus: bool = True, *args, **kwargs) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc_alpha = nn.Linear(n_hidden, 1)
        self.fc_beta = nn.Linear(n_hidden, 1)
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
    
if __name__ == "__main__":
    
    
    x = np.random.uniform(0, 1, 100)
    x = torch.from_numpy(x).view(-1,1).float()
    model = PredictorModel()
    out_probs = model(x)
    print(out_probs.shape)
