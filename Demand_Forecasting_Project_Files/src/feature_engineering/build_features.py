import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def build_features(sales_df, season_df):
    # 1. Clean and Merge
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    season_df['date'] = pd.to_datetime(season_df['date'])

    df = pd.merge(sales_df, season_df[['date', 'season', 'avg_temperature_c', 'event_type']], 
                  on='date', how='left')

    # 2. Advanced Feature Engineering (The Error Reducers)
    # Sort for time-series consistency
    df = df.sort_values(['product_name', 'region', 'date'])

    # Rolling average (7-day window)
    df['rolling_mean'] = df.groupby(['product_name', 'region'])['units_sold'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Momentum: Difference between today and yesterday
    df['diff_1'] = df.groupby(['product_name', 'region'])['units_sold'].diff().fillna(0)

    # 3. Cyclical Time Encoding
    # Converts 1-12 month into a circle so 12 is near 1
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)

    df['day_of_week'] = df['date'].dt.weekday
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7.0)

    # 4. Handle Categoricals
    cat_cols = ['product_name', 'category', 'region', 'flu_alert_level', 'season', 'event_type']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # 5. Final Cleaning
    # Forward fill then backward fill to catch any gaps from the merge
    df = df.ffill().bfill().fillna(0)

    # 6. Scaling with StandardScaler (better for MAPE than MinMaxScaler)
    num_cols = [
        'units_sold', 'unit_price', 'market_trend_index', 'economic_index', 
        'avg_temperature_c', 'rolling_mean', 'diff_1',
        'month_sin', 'month_cos', 'day_sin', 'day_cos'
    ]
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler, encoders