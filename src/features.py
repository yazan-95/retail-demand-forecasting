import numpy as np
import pandas as pd

def create_features(df):
    df = df.copy()

    # --- Ensure datetime ---
    df['date'] = pd.to_datetime(df['date'])

    # --- Sort ---
    df = df.sort_values(['product_id', 'date'])

    # --- Time features ---
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    # --- Lag features ---
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby('product_id')['sales'].shift(lag)

    # --- Rolling features ---
    df['rolling_mean_7'] = (
        df.groupby('product_id')['sales']
        .shift(1)
        .rolling(7)
        .mean()
    )

    df['rolling_std_7'] = (
        df.groupby('product_id')['sales']
        .shift(1)
        .rolling(7)
        .std()
    )

    # --- Momentum ---
    df['momentum'] = df['lag_1'] - df['lag_7']

    # --- Price change ---
    df['price_change'] = df.groupby('product_id')['price'].pct_change()

    # --- Promo last week ---
    df['promo_last_week'] = df.groupby('product_id')['promotion'].shift(7)

    return df