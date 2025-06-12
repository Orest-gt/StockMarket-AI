import torch

def create_dataset(data: torch.Tensor, window=120):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i]) # it was wrong and chatgpt saved it
        y.append(data[i, 3])
    return torch.tensor(X).clone().detach(), torch.tensor(y).clone().detach()
