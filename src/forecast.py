import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import time


# =========================
# FEATURE ENGINEERING (V2)
# =========================
def create_features(df):
    df = df.copy()

    df = df.rename(columns={

        'item_id': 'product_id',

        'promo': 'promotion'

    })

    df['date'] = pd.to_datetime(df['date'])

    if 'store_id' not in df.columns:
        df['store_id'] = 0

    df = df.sort_values(['store_id', 'product_id', 'date'])

    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    group = df.groupby(['store_id', 'product_id'])

    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = group['sales'].shift(lag)

    base = group['sales'].shift(1)

    for w in [7, 14, 28]:
        df[f'rolling_mean_{w}'] = base.rolling(w, min_periods=3).mean()
        df[f'rolling_std_{w}'] = base.rolling(w, min_periods=3).std()

    df['trend_7_14'] = df['rolling_mean_7'] - df['rolling_mean_14']
    df['trend_7_28'] = df['rolling_mean_7'] - df['rolling_mean_28']

    df['velocity'] = df['lag_1'] - df['lag_7']
    df['acceleration'] = (df['lag_1'] - df['lag_7']) - (df['lag_7'] - df['lag_14'])

    df['price_change'] = (
        group['price']
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    df['price_vs_avg'] = df['price'] / (group['price'].shift(1).rolling(7).mean())

    df['promo_last_week'] = group['promotion'].shift(7).fillna(0)
    df['promo_rolling_7'] = group['promotion'].shift(1).rolling(7).sum()

    df['product_avg'] = (
        df.groupby('product_id')['sales']
        .shift(1)
        .rolling(28)
        .mean()
    )

    df['store_avg'] = (
        df.groupby('store_id')['sales']
        .shift(1)
        .rolling(28)
        .mean()
    )

    df['global_avg'] = df['sales'].shift(1).rolling(28).mean()

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =========================
# FORECAST ENGINE
# =========================
def forecast(df, model, features, horizon=7):
    df = df.copy()

    # ✅ FIX 1: Normalize schema BEFORE anything
    df = df.rename(columns={
        'item_id': 'product_id',
        'promo': 'promotion'
    })

    df['date'] = pd.to_datetime(df['date'])

    if 'store_id' not in df.columns:
        df['store_id'] = 0

    last_date = df['date'].max()
    print("Last available date:", last_date)

    # ✅ FIX 2: Use ONLY latest snapshot per series
    latest_df = (
        df.sort_values(['store_id', 'product_id', 'date'])
        .groupby(['store_id', 'product_id'])
        .tail(1)
        [['store_id', 'product_id', 'price', 'promotion']]
        .copy()
    )

    results = []

    for step in range(1, horizon + 1):
        print(f"\nForecasting step {step}/{horizon}")

        future_date = last_date + pd.Timedelta(days=step)

        future_df = latest_df.copy()
        future_df['date'] = future_date
        future_df['sales'] = np.nan

        df = pd.concat([df, future_df], ignore_index=True)

        df = create_features(df)

        current = df[df['date'] == future_date].copy()

        X = current.drop(columns=['sales', 'date'])

        for col in ['product_id', 'store_id']:
            if col in X.columns:
                X[col] = X[col].astype('category').cat.codes

        X = X[features]

        preds = model.predict(X)
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)

        df.loc[df['date'] == future_date, 'sales'] = preds

        current['prediction'] = preds
        results.append(current[['store_id', 'product_id', 'date', 'prediction']])

    forecast_df = pd.concat(results)

    return forecast_df


# =========================
# MAIN
# =========================
def main():

    BASE_DIR = Path(__file__).resolve().parent.parent

    model_path = BASE_DIR / "notebooks" / "lgbm_sales_model_v2.pkl"
    artifact = joblib.load(model_path)

    model = artifact["model"]
    features = artifact["features"]

    print("Model loaded.")

    data_path = BASE_DIR / "data" / "retail_sales.csv"
    df = pd.read_csv(data_path)

    print("Data loaded:", df.shape)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    forecast_df = forecast(df, model, features, horizon=7)

    print("\nForecast sample:")
    print(forecast_df.head())

    save_path = BASE_DIR / "data" / "forecast_output.csv"
    forecast_df.to_csv(save_path, index=False)

    print("Forecast saved to:", save_path)


if __name__ == "__main__":
    main()