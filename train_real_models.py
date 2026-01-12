import pandas as pd
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder

def train_real_models():
    print("🚀 Starting real model training...")
    
    try:
        data_1 = pd.read_csv('McKesson_Large_Demand_Forecasting_Dataset.csv')
        data_2 = pd.read_csv('daily_dataset_2023.csv')
    except FileNotFoundError:
        print("❌ Error: CSV files not found!")
        return

    # --- THE FIX: Clean the product names in the CSV ---
    # This removes spaces and replaces them with underscores to match your API
    data_1['product_name'] = data_1['product_name'].str.replace(' ', '_').str.replace('-', '_')
    
    # Let's print the names found in the CSV to be sure
    print("Products found in CSV:", data_1['product_name'].unique())

    data_1['month_day'] = data_1['date'].str[5:10]
    data_2['month_day'] = data_2['date'].str[5:10]
    df = pd.merge(data_1, data_2[['month_day', 'season']], on='month_day', how='left')
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['week_of_month'] = df['date'].apply(lambda d: (d.day-1)//7 + 1)
    
    df = df.sort_values(['product_name', 'date'])
    df['lag_1'] = df.groupby('product_name')['units_sold'].shift(1)
    df['lag_7'] = df.groupby('product_name')['units_sold'].shift(7)
    df['roll_mean_7'] = df.groupby('product_name')['units_sold'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['roll_std_7'] = df.groupby('product_name')['units_sold'].transform(lambda x: x.shift(1).rolling(7).std())
    df.dropna(inplace=True)

    le = LabelEncoder()
    df['season_num'] = le.fit_transform(df['season'].astype(str))
    df['flu_num'] = le.fit_transform(df['flu_alert_level'].astype(str))

    features = ['flu_num', 'season_num', 'week_of_month', 'month', 
                'lag_1', 'lag_7', 'roll_mean_7', 'roll_std_7']

    meds_config = {
        "Amoxicillin_500mg": ("rf_model_amoxicillin.joblib", RandomForestRegressor(n_estimators=100)),
        "Atorvastatin_20mg": ("rf_model_atorvastatin.joblib", RandomForestRegressor(n_estimators=100)),
        "Insulin_Glargine_10ml": ("lgb_model_insulin_glargine.joblib", LGBMRegressor(verbose=-1)),
        "Surgical_Gloves_Box": ("lgb_model_surgical_gloves_box.joblib", LGBMRegressor(verbose=-1)),
        "Surgical_Masks_Box": ("xg_model_surgical_masks_box.joblib", XGBRegressor())
    }

    for med_name, (filename, model) in meds_config.items():
        # Searching for the name
        subset = df[df['product_name'] == med_name]
        
        if not subset.empty:
            print(f"⌛ Training {med_name} (Rows: {len(subset)})...")
            X = subset[features]
            y = subset['units_sold']
            model.fit(X, y)
            dump(model, filename)
            print(f"✅ Saved: {filename}")
        else:
            print(f"⚠️ Could not find '{med_name}' in CSV. Check spelling!")

if __name__ == "__main__":
    train_real_models()