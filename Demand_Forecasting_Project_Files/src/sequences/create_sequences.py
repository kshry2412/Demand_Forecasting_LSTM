import numpy as np



def create_lstm_sequences(df, feature_cols, target_col, lookback=30):

    X, y = [], []



    for i in range(lookback, len(df)):

        X.append(df[feature_cols].iloc[i-lookback:i].values)

        y.append(df[target_col].iloc[i])



    return np.array(X), np.array(y)