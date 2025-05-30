def inverse_transform_close_only(scaled_close, scaler, close_idx=3):
    import numpy as np

    if not hasattr(scaler, 'center_'):
        raise ValueError("Scaler is not fitted yet.")

    scaled_close = scaled_close.reshape(-1)
    dummy = np.zeros((len(scaled_close), len(scaler.center_)))  # π.χ. (3401, 5)
    dummy[:, close_idx] = scaled_close
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, close_idx].reshape(-1, 1)