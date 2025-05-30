import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from make_dataset import create_dataset
from dataset_finishing import dataset_finish
from configs import wait, patience, best_val_loss, num_epochs, add_training_date_start, add_training_data_end, best_model_path, best_model_path_end, ticker
import yfinance as yf
import numpy as np
from model import Model
from live import live_graph_training
from regular import regular_training

exact_model = input("Search for the exact model date: ")

device = "cuda" if torch.cuda.is_available() else "cpu"

data = yf.download(ticker, start=add_training_date_start, end=add_training_data_end, auto_adjust=True) # logiko
features = data[["Open", "High", "Low", "Close", "Volume"]].values # bazoume allo ena dim gia sostotita stis kontines times. Γιατί; Οι scaler & NN θέλουν 2D inputs.

scaler = RobustScaler()
scaled_data = scaler.fit_transform(features)
def inverse_transform_close_only(scaled_close, scaler_=scaler, close_idx=3):

    if not hasattr(scaler, 'center_'):
        raise ValueError("Scaler is not fitted yet.")

    scaled_close = scaled_close.reshape(-1)
    dummy = np.zeros((len(scaled_close), len(scaler_.center_)))  # π.χ. (3401, 5)
    dummy[:, close_idx] = scaled_close
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, close_idx].reshape(-1, 1)

X, y = create_dataset(scaled_data, window=30)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
X, y = X.to(device), y.to(device)

model = Model()
model = model.to(device)

loss_tool = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#checkpoint_path = os.path.join(best_model_path)
checkpoint = torch.load(best_model_path + exact_model + best_model_path_end)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model = model.to(device)
train_loader, val_loader, loader = dataset_finish(X, y)

def train(live=True):
    if live:
        live_graph_training(X, y, inverse_transform_close_only, model, num_epochs, loader, device, loss_tool, optimizer, val_loader, best_val_loss, wait, patience)
    else:
        regular_training(X, y, inverse_transform_close_only, model, num_epochs, loader, device, loss_tool, optimizer, val_loader, best_val_loss, wait, patience)

if __name__ == "__main__":
    train()

# FINE TUNER
# Αυτό είναι αυτός ο κώδικας
