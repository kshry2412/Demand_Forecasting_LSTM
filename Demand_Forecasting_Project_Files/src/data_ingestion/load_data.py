import pandas as pd

def load_raw_data(sales_path, seasonal_path):
    sales_df = pd.read_csv(sales_path)
    season_df = pd.read_csv(seasonal_path)
    return sales_df, season_df
