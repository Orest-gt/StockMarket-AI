import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from live import live_graph_training
from regular import regular_training
from model import Model
from make_dataset import create_dataset
from dataset_finishing import dataset_finish
from configs import num_epochs, ticker, wait, patience, best_val_loss, main_training_date_start, main_training_date_end

debugging_print = False
on_tpu = False
device = None

if not on_tpu:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    print("TPU not supported!")
    device = "cuda" if torch.cuda.is_available() else "cpu"


data = yf.download(ticker, start=main_training_date_start, end=main_training_date_end, auto_adjust=True) # logiko
features = data[["Open", "High", "Low", "Close", "Volume"]].values # bazoume allo ena dim gia sostotita stis kontines times. Γιατί; Οι scaler & NN θέλουν 2D inputs.

scaler = RobustScaler()
scaled_data = scaler.fit_transform(features)

def inverse_transform_close_only(scaled_close, scaler_=scaler, close_idx=3):
    import numpy as np

    if not hasattr(scaler, 'center_'):
        raise ValueError("Scaler is not fitted yet.")

    scaled_close = scaled_close.reshape(-1)
    dummy = np.zeros((len(scaled_close), len(scaler_.center_)))  # π.χ. (3401, 5)
    dummy[:, close_idx] = scaled_close
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, close_idx].reshape(-1, 1)

X, y = create_dataset(scaled_data)
X, y = X.to(device), y.to(device)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # add 2d dim

model = Model()
model = model.to(device)
loss_tool = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader, val_loader, loader = dataset_finish(X, y)

def train(live=False):
    if live:
        live_graph_training(X, y, inverse_transform_close_only, model, num_epochs, loader, device, loss_tool, optimizer, val_loader, best_val_loss, wait, patience)
    else:
        regular_training(X, y, inverse_transform_close_only, model, num_epochs, loader, device, loss_tool, optimizer, val_loader, best_val_loss, wait, patience)


if __name__ == "__main__":
    train(False)