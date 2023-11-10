import numpy as np
import torch

from torch.utils.data import Dataset

def sine_fct_prediction(x, freq: float = 10.0):

    pred_fct = lambda x: 0.5 * np.sin(freq * x) + .5
    return pred_fct(x)

def generate_bernoulli_labels(x_inst: np.ndarray, fct_pred):    

    preds = fct_pred(x_inst)

    # sample labels from multinomial distribution
    labels = np.random.binomial(1, preds)

    return labels, preds

class BernoulliSineDataset(Dataset):

    def __init__(self, n_samples: int, sine_factor: int = 5) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.sine_factor = sine_factor
        self.x_inst = torch.from_numpy(np.random.uniform(0,1, n_samples)).float()
        fct_pred = lambda x: sine_fct_prediction(x, freq=self.sine_factor)
        self.y_labels, self.preds = generate_bernoulli_labels(self.x_inst, fct_pred)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        
        return self.x_inst[index], self.y_labels[index]
    
class BernoulliSineDatasetsplitted(Dataset):

    def __init__(self, n_samples_1: int, n_samples_2: int, sine_factor: int = 5, split: float=0.5):
        super().__init__()
        self.n_samples_1 = n_samples_1
        self.n_samples_2 = n_samples_2
        self.sine_factor = sine_factor
        self.split = split
        self.x_inst_1 = torch.from_numpy(np.random.uniform(0,split, n_samples_1)).float()
        self.x_inst_2 = torch.from_numpy(np.random.uniform(split,1, n_samples_2)).float()
        self.x_inst = torch.cat((self.x_inst_1, self.x_inst_2), dim=0)
        fct_pred = lambda x: sine_fct_prediction(x, freq=self.sine_factor)
        self.y_labels, self.preds = generate_bernoulli_labels(self.x_inst, fct_pred)

    def __len__(self):
        return self.n_samples_1 + self.n_samples_2
    
    def __getitem__(self, index):
            
            return self.x_inst[index], self.y_labels[index]


class SineRegressionDataset(Dataset):

    def __init__(self, n_samples: int, sine_factor: int = 5, x_max: float = 1.0,
                 n_samples_2: int = 0, eps_var: float = 0.) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.sine_factor = sine_factor
        self.x_inst_in = torch.from_numpy(np.random.uniform(0,x_max, n_samples)).float()
        self.x_inst_out = torch.from_numpy(np.random.uniform(x_max, 1, n_samples_2)).float()
        eps = torch.normal(torch.zeros(n_samples + n_samples_2), eps_var)
        if n_samples_2 > 0:
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
    fct_preds = sine_fct_prediction

    x_inst = np.random.uniform(0,1,500)
    preds = sine_fct_prediction(x_inst)
    print(preds)
    labels, preds = generate_bernoulli_labels(x_inst, fct_preds)
    print(labels)

    import matplotlib.pyplot as plt
    print(x_inst.shape)
    print(labels.shape)
    fig, ax = plt.subplots()
    ax.scatter(x_inst, labels)
    ax.scatter(x_inst, preds)
    plt.show()

    dataset = BernoulliSineDatasetsplitted(100, 10)
    print(len(dataset))
    print(len(dataset.x))
    print(len(dataset.y_labels))
    print(len(dataset.preds))