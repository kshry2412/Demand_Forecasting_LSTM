from fastapi import FastAPI
from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import date
from dateutil.relativedelta import relativedelta
from joblib import load
import numpy as np

app = FastAPI()

# --- LOAD MODELS ONCE AT STARTUP ---
try:
    MODELS = {
        "Amoxicillin_500mg": load("rf_model_amoxicillin.joblib"),
        "Atorvastatin_20mg": load("rf_model_atorvastatin.joblib"),
        "Insulin_Glargine_10ml": load("lgb_model_insulin_glargine.joblib"),
        "Surgical_Gloves_Box": load("lgb_model_surgical_gloves_box.joblib"),
        "Surgical_Masks_Box": load("xg_model_surgical_masks_box.joblib")
    }
    print("✅ All models loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}. Ensure you ran train_real_models.py")

def calculation_logic(input1: str, input2: int):
    # 1. Load Data
    data_1 = pd.read_csv('McKesson_Large_Demand_Forecasting_Dataset.csv')
    data_2 = pd.read_csv('daily_dataset_2023.csv')
    
    # 2. Basic Cleaning
    data_1.drop(columns=['product_id','region'], inplace=True, errors='ignore')
    data_1['month_day'] = data_1['date'].str[5:10]
    data_2['month_day'] = data_2['date'].str[5:10]
    
    merged_df = pd.merge(data_1, data_2[['month_day', 'season']], on='month_day', how='left')
    merged_df['month_And_day_of_month'] = merged_df['month_day']
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # 3. Feature Engineering
    merged_df['month_day_int'] = merged_df['month_day'].str[3:5].astype(int)
    merged_df['day_of_week'] = merged_df['date'].dt.weekday
    merged_df['week_of_month'] = merged_df['date'].apply(lambda d: (d.day-1)//7 + 1)
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['lag_1'] = merged_df['units_sold'].shift(1) 
    merged_df['lag_7'] = merged_df['units_sold'].shift(7)
    merged_df['roll_mean_7'] = merged_df['units_sold'].shift(1).rolling(7).mean()
    merged_df['roll_std_7'] = merged_df['units_sold'].shift(1).rolling(7).std()
    merged_df.dropna(inplace=True)

    # 4. Encoding
    le = LabelEncoder()
    # We need to find the specific ID for the medicine requested
    merged_df['product_id_encoded'] = le.fit_transform(merged_df['product_name'].astype(str))
    prod_map = dict(zip(le.classes_, range(len(le.classes_))))
    
    # Clean up the names in the map to handle underscores vs spaces
    clean_prod_map = {k.replace(' ', '_').replace('-', '_'): v for k, v in prod_map.items()}
    target_id = clean_prod_map.get(input1, 0)

    # Encode other categories for the dataframe
    for col in ['category','flu_alert_level','season']:
        merged_df[col] = le.fit_transform(merged_df[col].astype(str))

    # 5. Price Configuration
    prices = {
        "Amoxicillin_500mg": 1.25, "Atorvastatin_20mg": 0.85, 
        "Insulin_Glargine_10ml": 2.50, "Surgical_Gloves_Box": 1.75, "Surgical_Masks_Box": 1.10
    }
    price = prices.get(input1, 1.0)

    # 6. Time Calculation
    today = date.today()
    future_date = today + relativedelta(months=+input2)
    no_of_weeks = (future_date - today).days // 7
    
    # 7. Create Historical Averages (Filtered by the CORRECT product)
    new_dataset = merged_df.loc[merged_df['product_id_encoded'] == target_id]
    medicine_averagedf = new_dataset.groupby(['month', 'week_of_month']).agg({
        'lag_1': 'mean', 'lag_7': 'mean', 'roll_mean_7': 'mean', 'roll_std_7': 'mean'
    }).reset_index()

    total_count = 0
    date_cursor = today

    # 8. Prediction Loop
    for i in range(no_of_weeks):
        week_stats = {"flu": [], "season": [], "w_month": [], "month": [], "l1": [], "l7": [], "rm7": [], "rs7": []}
        
        for j in range(7):
            date_cursor += relativedelta(days=+1)
            fmt_date = date_cursor.strftime("%Y-%m-%d")
            mm_dd = fmt_date[5:10]
            curr_month = int(fmt_date[5:7])
            curr_w_month = (int(fmt_date[8:10])-1)//7 + 1
            
            # Fetch daily context
            day_data = merged_df.loc[merged_df['month_And_day_of_month'] == mm_dd]
            if not day_data.empty:
                week_stats["flu"].append(day_data['flu_alert_level'].iloc[0])
                week_stats["season"].append(day_data['season'].iloc[0])
            else:
                week_stats["flu"].append(0)
                week_stats["season"].append(0)
            
            # Fetch historical averages
            avg_data = medicine_averagedf.loc[(medicine_averagedf['month'] == curr_month) & (medicine_averagedf['week_of_month'] == curr_w_month)]
            if not avg_data.empty:
                week_stats["l1"].append(avg_data['lag_1'].iloc[0])
                week_stats["l7"].append(avg_data['lag_7'].iloc[0])
                week_stats["rm7"].append(avg_data['roll_mean_7'].iloc[0])
                week_stats["rs7"].append(avg_data['roll_std_7'].iloc[0])
            
            week_stats["w_month"].append(curr_w_month)
            week_stats["month"].append(curr_month)

        # Build feature vector [flu, season, week_month, month, lag1, lag7, mean7, std7]
        feat = [
            np.mean(week_stats["flu"]), 
            np.mean(week_stats["season"]),
            max(set(week_stats["w_month"]), key=week_stats["w_month"].count),
            max(set(week_stats["month"]), key=week_stats["month"].count),
            np.mean(week_stats["l1"] if week_stats["l1"] else [0]),
            np.mean(week_stats["l7"] if week_stats["l7"] else [0]),
            np.mean(week_stats["rm7"] if week_stats["rm7"] else [0]),
            np.mean(week_stats["rs7"] if week_stats["rs7"] else [0])
        ]

        model = MODELS.get(input1)
        if model:
            # PRED IS RAW UNITS: No inv_boxcox or expm1 needed
            prediction = model.predict([feat])
            pred_val = max(0, float(prediction[0])) # Ensure no negative demand
            total_count += (pred_val * 7)

    return total_count, total_count * price

@app.get("/process")
async def process_data(input1: str, input2: int):
    output1, output2 = calculation_logic(input1, input2)
    return {
        "medicine": input1,
        "forecast_period_months": input2,
        "units_forecasted": round(output1, 2), 
        "estimated_revenue": round(output2, 2)
    }