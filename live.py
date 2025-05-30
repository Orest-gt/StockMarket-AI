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
#folder_path = best_model_path + now
#os.makedirs(folder_path, exist_ok=True)

def live_graph_training(X: Tensor, y: Tensor, inverse_transform_close_only, model: Model, num_epochs: int, loader, device: str, loss_tool, optimizer, val_loader, best_val_loss: float, wait: int, patience: int) -> None:
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(y))
    line_real, = ax.plot(x, inverse_transform_close_only(y.view(-1, 1)), label="REAL", color='blue')
    line_pred, = ax.plot(x, inverse_transform_close_only(y.view(-1, 1)), label="PREDICTED", color='orange')

    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Training progress (LIVE)")
    ax.legend()

    for epoch in range(num_epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        model.train()
        for batch_X, batch_y in progress:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = loss_tool(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update γραφήματος κάθε batch (αν και επιβαρύνει την ταχύτητα)

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

        model.eval()
        with torch.no_grad():
            pred = model(X).detach().cpu().numpy()

        real_ones = inverse_transform_close_only(y.cpu().numpy().reshape(-1, 1))
        predicted_prices = inverse_transform_close_only(pred.reshape(-1, 1))

        line_real.set_ydata(real_ones)
        line_pred.set_ydata(predicted_prices)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()  # Πολύ μικρότερο delay