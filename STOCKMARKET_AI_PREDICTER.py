import yfinance as yf
import torch
import numpy as np
from model import Model
from sklearn.preprocessing import RobustScaler
from make_dataset import create_dataset
from configs import best_model_path, predict_data_period, ticker, best_model_path_end
from yfinance import Ticker

#PEPE24478-USD
device = "cuda" if torch.cuda.is_available() else "cpu"

# Φορτώνουμε μοντέλο + optimizer
model = Model()
optimizer = torch.optim.Adam(model.parameters())

exact_model = input("Search for the exact model date: ")
checkpoint = torch.load(best_model_path + exact_model + best_model_path_end)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

ticker_ = Ticker(ticker)

data = ticker_.history(period=predict_data_period)
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

# Η συνάρτηση πρόβλεψης
def predict_future(model: Model, last_window, n_days=10):
    preds = []
    window = last_window.clone()

    for _ in range(n_days):
        with torch.no_grad():
            pred = model(window.unsqueeze(0))  # (1, 60, 5)
            preds.append(pred.cpu().item())

            new_entry = torch.zeros(window.shape[1])
            new_entry[3] = pred  # μόνο Close

            window = torch.cat((window[1:], new_entry.unsqueeze(0)), dim=0)

    preds = np.array(preds).reshape(-1, 1)
    return inverse_transform_close_only(preds)

# Παράδειγμα χρήσης
last_3_days = torch.tensor(scaled_data[-3:], dtype=torch.float32)
future_prices = predict_future(model, last_3_days, n_days=10)

print("Πρόβλεψη 10 ημερών (Close):")
print(future_prices)

def trading_signal(predictions, current_price, buy_threshold=0.02, sell_threshold=-0.02, ma_window=3) -> str:
    print("\n")
    # predictions: numpy array με προβλέψεις 10 ημερών (Close prices)
    # current_price: η τελευταία γνωστή τιμή Close
    # buy_threshold: το ποσοστό αύξησης για σήμα αγοράς (πχ 2%)
    # sell_threshold: το ποσοστό πτώσης για σήμα πώλησης (πχ -2%)
    # ma_window: παράθυρο για μέση τιμή κινητής μέσης

    # Προσθέτουμε την τρέχουσα τιμή στην αρχή
    prices = np.insert(predictions.flatten(), 0, current_price)

    # Υπολογίζουμε returns (ποσοστιαία αλλαγή) μέρα με τη μέρα
    returns = (prices[1:] - prices[:-1]) / prices[:-1]

    # Υπολογίζουμε κινητό μέσο όρο των προβλέψεων (για τις 10 μέρες)
    moving_avg = np.convolve(prices[1:], np.ones(ma_window) / ma_window, mode='valid')

    # Κανόνας 1: Αν το μέσο όρο των returns είναι πάνω από buy_threshold → αγορά
    if returns.mean() > buy_threshold and moving_avg[-1] > current_price:
        return "Αγόρασε\n"
    # Κανόνας 2: Αν το μέσο όρο των returns είναι κάτω από sell_threshold → πώλησε
    elif returns.mean() < sell_threshold and moving_avg[-1] < current_price:
        return "Πούλησε\n"
    else:
        return "Κράτα\n"

# Παράδειγμα χρήσης:
predictions = np.array(future_prices)

ticker = yf.Ticker(ticker)
current_price = data['Close'].iloc[-1]
print(trading_signal(predictions, current_price))