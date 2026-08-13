import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path


# =========================
# 1. FEATURE ENGINEERING
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

    # =========================
    # TIME FEATURES
    # =========================
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    group = df.groupby(['store_id', 'product_id'])

    # =========================
    # LAGS
    # =========================
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = group['sales'].shift(lag)

    base = group['sales'].shift(1)

    # =========================
    # ROLLING WINDOWS
    # =========================
    for w in [7, 14, 28]:
        df[f'rolling_mean_{w}'] = base.rolling(w, min_periods=3).mean()
        df[f'rolling_std_{w}'] = base.rolling(w, min_periods=3).std()

    # =========================
    # TRENDS
    # =========================
    df['trend_7_14'] = df['rolling_mean_7'] - df['rolling_mean_14']
    df['trend_7_28'] = df['rolling_mean_7'] - df['rolling_mean_28']

    # =========================
    # VELOCITY / ACCELERATION
    # =========================
    df['velocity'] = df['lag_1'] - df['lag_7']
    df['acceleration'] = (df['lag_1'] - df['lag_7']) - (df['lag_7'] - df['lag_14'])

    # =========================
    # PRICE FEATURES
    # =========================
    df['price_change'] = (
        group['price']
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    df['price_vs_avg'] = (
        df['price'] /
        (group['price'].shift(1).rolling(7).mean())
    )

    # =========================
    # PROMO FEATURES
    # =========================
    df['promo_last_week'] = group['promotion'].shift(7).fillna(0)
    df['promo_rolling_7'] = group['promotion'].shift(1).rolling(7).sum()

    # =========================
    # HIERARCHICAL FEATURES (LEAKAGE SAFE)
    # =========================
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

    # =========================
    # NUMERICAL STABILITY
    # =========================
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =========================
# 2. ERROR ANALYSIS
# =========================
def error_analysis(df, preds):
    df = df.copy()

    df['prediction'] = preds
    df['error'] = df['sales'] - df['prediction']
    df['abs_error'] = df['error'].abs()

    print("\n=========================")
    print("ERROR ANALYSIS")
    print("=========================")

    print("MAE:", df['abs_error'].mean())

    print("\nWorst cases:")
    print(df.sort_values('abs_error', ascending=False).head(20)[
        ['product_id', 'date', 'sales', 'prediction', 'abs_error']
    ])

    print("\nWorst products:")
    print(
        df.groupby('product_id')['abs_error']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nWorst stores:")
    print(
        df.groupby('store_id')['abs_error']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    return df


# =========================
# 3. MAIN PIPELINE
# =========================
def main():

    BASE_DIR = Path(__file__).resolve().parent.parent

    # 🔴 LOAD V2 MODEL
    model_path = BASE_DIR / "notebooks" / "lgbm_sales_model_v2.pkl"
    artifact = joblib.load(model_path)

    model = artifact["model"]
    target_transform = artifact.get("target_transform", "log1p")

    print("Model loaded from:", model_path)

    # Features list
    INVALID_COLUMNS = ['sales', 'date']

    expected_features = [
        f for f in artifact["features"]
        if f not in INVALID_COLUMNS and f not in ['', None]
    ]

    expected_features = list(dict.fromkeys(expected_features))
    print("Expected features:", len(expected_features))

    # Load data
    data_path = BASE_DIR / "data" / "retail_sales.csv"
    df = pd.read_csv(data_path)

    print("Data loaded:", df.shape)

    # Schema fixes
    if 'item_id' in df.columns and 'product_id' not in df.columns:
        df = df.rename(columns={'item_id': 'product_id'})

    if 'promotion' not in df.columns:
        print("WARNING: 'promotion' missing → filling with 0")
        df['promotion'] = 0

    if 'store_id' not in df.columns:
        print("WARNING: 'store_id' missing → filling with 0")
        df['store_id'] = 0

    if 'promo' in df.columns:
        df = df.drop(columns=['promo'])

    # Validation
    required_cols = ['date', 'product_id', 'sales', 'price', 'promotion', 'store_id']
    missing_raw = set(required_cols) - set(df.columns)

    if missing_raw:
        raise ValueError(f"Missing raw columns: {missing_raw}")

    # Datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Feature engineering
    df = create_features(df)
    print("Features created.")

    before = df.shape
    df = df.dropna()
    print(f"Dropna: {before} → {df.shape}")

    # Build X
    X = df.drop(columns=['sales', 'date'], errors='ignore').copy()

    for col in ['product_id', 'store_id']:
        if col in X.columns:
            X[col] = X[col].astype('category').cat.codes

    # Validation
    missing = set(expected_features) - set(X.columns)
    extra = set(X.columns) - set(expected_features)

    print("Missing:", missing)
    print("Extra:", extra)

    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = X[expected_features]

    if X.isnull().any().any():
        raise ValueError("NaNs detected")

    print("Feature alignment successful.")

    # Predict
    print("\nStarting prediction...")
    start = time.time()

    preds = model.predict(X)

    end = time.time()
    print(f"Prediction done in {end - start:.2f} seconds")

    print("\n=== Prediction Stats (RAW) ===")
    print("Min:", preds.min())
    print("Max:", preds.max())
    print("Mean:", preds.mean())

    # Transform target
    if target_transform == "log1p":
        preds = np.expm1(preds)
    elif target_transform == "log":
        preds = np.exp(preds)

    preds = np.clip(preds, 0, None)

    print("\n=== Prediction Stats (FINAL) ===")
    print("Min:", preds.min())
    print("Max:", preds.max())
    print("Mean:", preds.mean())

    print("Prediction completed.")

    # Error analysis
    error_analysis(df, preds)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()