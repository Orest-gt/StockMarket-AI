import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int= 64, num_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
            )
    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.mlp(output)
        return output

'''
: σημαίνει "όλα τα samples στο batch"
-1 σημαίνει "το τελευταίο στοιχείο της ακολουθίας" (τελευταίο timestep)
: σημαίνει "όλα τα features του κρυφού state" (hidden_size)
'''