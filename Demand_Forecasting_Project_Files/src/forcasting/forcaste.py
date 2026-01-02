import numpy as np

def forecast_future_demand(model, X_last, forecast_days=30):

    future_predictions = []

    current_sequence = X_last[-1:]  # Get the last sequence from test data

   

    for _ in range(forecast_days):

        pred = model.predict(current_sequence)

        future_predictions.append(pred[0,0])

       

        # Update the sequence for the next prediction

        current_sequence = np.roll(current_sequence, -1, axis=1)

        current_sequence[0, -1, 0] = pred  # Set the predicted value to the last position

   

    return future_predictions