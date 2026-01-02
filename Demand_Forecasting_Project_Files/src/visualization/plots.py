import matplotlib.pyplot as plt
import seaborn as sns


def plot_historical_demand(df, product_id=0):
  
    product_df = df[df['product_name'] == product_id]
   
    plt.figure(figsize=(12, 5)) 
    plt.plot(product_df['date'], product_df['units_sold'])
    plt.title("Historical Demand Over Time")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.show()


def plot_seasonality(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='season', y='units_sold', data=df)
    plt.title("Demand Distribution by Season")
    plt.show()


def plot_flu_impact(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x='flu_alert_level',
        y='units_sold',
        alpha=0.6
    )
    plt.title("Impact of Flu Alert on Demand")
    plt.show()


def plot_lstm_sample(X, feature_index=0):
    plt.figure(figsize=(8, 4))
    plt.plot(X[0, :, feature_index])
    plt.title("Sample LSTM Input Sequence")
    plt.xlabel("Time Steps")
    plt.ylabel("Scaled Feature Value")
    plt.show()


def plot_train_test_split(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)

    plt.figure(figsize=(12, 5))
    plt.plot(df['date'][:split], df['units_sold'][:split], label='Train')
    plt.plot(df['date'][split:], df['units_sold'][split:], label='Test')
    plt.legend()
    plt.title("Train-Test Time Split")
    plt.show()
