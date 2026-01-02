import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_lstm_model(model, X, y, scaler):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        preds = model(inputs).numpy()

    # 1. Reverse the Scaling
    num_features = scaler.n_features_in_
    dummy_preds = np.zeros((len(preds), num_features))
    dummy_actual = np.zeros((len(y), num_features))

    dummy_preds[:, 0] = preds.flatten()
    dummy_actual[:, 0] = y.flatten()

    inv_preds = scaler.inverse_transform(dummy_preds)[:, 0]
    inv_actual = scaler.inverse_transform(dummy_actual)[:, 0]

    # 2. Calculate Metrics
    mae = mean_absolute_error(inv_actual, inv_preds)
    mape = np.mean(np.abs((inv_actual - inv_preds) / (inv_actual + 1))) * 100

    print(f"\n--- Final Results ---")
    print(f"MAE: {mae:.2f} units")
    print(f"MAPE: {mape:.2f}%")

    # 3. Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(inv_actual[-100:], label='Actual Demand', color='blue', linewidth=2)
    plt.plot(inv_preds[-100:], label='Predicted Demand', color='orange', linestyle='--', linewidth=2)
    plt.title('Demand Forecasting: Actual vs Predicted (Last 100 Days)')
    plt.xlabel('Time Steps')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return inv_actual, inv_preds