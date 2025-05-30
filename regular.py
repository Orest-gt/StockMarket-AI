import torch
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from configs import best_model_path, best_model_path_end
from model import Model
from datetime import datetime

now = datetime.now()
now = now.strftime("%Y-%m-%d-%H-%M")

def regular_training(X: Tensor, y: Tensor, inverse_transform_close_only, model: Model, num_epochs: int, loader, device: str, loss_tool, optimizer: torch.optim.Adam, val_loader, best_val_loss: float, wait: int, patience: int) -> None:
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            # print(output.shape)
            # print(batch_y.shape)
            loss = loss_tool(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_output = model(val_x)
                # print(val_output.shape)
                val_loss = loss_tool(val_output, val_y)
                val_losses.append(val_loss.item())
        avg_val_loss = np.mean(val_losses)

        if epoch % 100 == 0 and epoch != 0:
            print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            # Save the best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss_state_dict': loss_tool.state_dict(),
            }, best_model_path + now + best_model_path_end)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
    with torch.no_grad():
        pred = model(X).cpu().numpy()

    real_ones = inverse_transform_close_only(y.view(-1, 1))
    predicted_prices = inverse_transform_close_only(pred)

    plt.figure(figsize=(12, 6))
    plt.plot(real_ones, label="REAL")
    plt.plot(predicted_prices, label="POSSIBLE")
    plt.title("FINANCE")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()