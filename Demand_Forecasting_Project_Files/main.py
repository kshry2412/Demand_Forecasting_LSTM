import os
import numpy as np
from src.data_ingestion.load_data import load_raw_data
from src.feature_engineering.build_features import build_features
from src.sequences.create_sequences import create_lstm_sequences
from src.training.train import train_lstm_model
from src.evaluation.evaluate_model import evaluate_lstm_model

# Config
LOOKBACK = 30
TARGET_COL = "units_sold"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_PATH = os.path.join(BASE_DIR, "data", "raw", "McKesson_Large_Demand_Forecasting_Dataset.csv")
SEASON_PATH = os.path.join(BASE_DIR, "data", "raw", "daily_dataset_2023.csv")

def main():
    sales_df, season_df = load_raw_data(SALES_PATH, SEASON_PATH)
    df, scaler, encoders = build_features(sales_df, season_df)

    # EXACT Match for new feature list
    feature_cols = [
    'product_name', 'category', 'region', 'unit_price',
    'market_trend_index', 'flu_alert_level', 'economic_index', 
    'season', 'event_type', 'avg_temperature_c', 
    'rolling_mean', 'diff_1',
    'month_sin', 'month_cos', 'day_sin', 'day_cos'
    ]

    X, y = create_lstm_sequences(df, feature_cols, TARGET_COL, LOOKBACK)
    
    # Check for NaNs before training
    if np.isnan(X).any() or np.isnan(y).any():
        print("CRITICAL: NaNs found in sequences. Filling with 0...")
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

    print(f"Ready to train on {X.shape[0]} samples with {X.shape[2]} features.")
    model = train_lstm_model(X, y, epochs=50) # Reduced epochs for testing
    evaluate_lstm_model(model, X, y, scaler)

if __name__ == "__main__":
    main()