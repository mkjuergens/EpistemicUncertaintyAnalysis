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



if __name__ == "__main__":
    fct_preds = sine_fct_prediction

    x_inst = np.random.uniform(0,1,500)
    preds = sine_fct_prediction(x_inst)
    print(preds)
    labels = generate_bernoulli_labels(x_inst, fct_preds)
    print(labels)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x_inst, labels)
    ax.scatter(x_inst, preds)
    plt.show()