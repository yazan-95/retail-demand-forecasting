import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from lightgbm import LGBMRegressor


# =========================
# FEATURE ENGINEERING (V2)
# MUST MATCH train.py EXACTLY
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
# MODEL (same as train.py)
# =========================
def build_model():
    return LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=255,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )


# =========================
# BACKTEST LOGIC
# =========================
def run_backtest(df, n_folds=3):

    print("\n=========================")
    print("BACKTEST START")
    print("=========================")

    df = df.sort_values("date")

    unique_dates = df['date'].sort_values().unique()

    fold_size = int(len(unique_dates) * 0.1)  # 10% per validation
    results = []

    for fold in range(n_folds):

        print(f"\n--- Fold {fold + 1} ---")

        val_start = len(unique_dates) - (n_folds - fold) * fold_size
        val_end = val_start + fold_size

        train_dates = unique_dates[:val_start]
        val_dates = unique_dates[val_start:val_end]

        train_df = df[df['date'].isin(train_dates)]
        val_df = df[df['date'].isin(val_dates)]

        print(f"Train size: {train_df.shape}, Val size: {val_df.shape}")

        # Features
        train_df = create_features(train_df)
        val_df = create_features(val_df)

        train_df = train_df.dropna()
        val_df = val_df.dropna()

        X_train = train_df.drop(columns=['sales', 'date'])
        y_train = np.log1p(train_df['sales'])

        X_val = val_df.drop(columns=['sales', 'date'])
        y_val = val_df['sales']

        # Encode categoricals
        for col in ['product_id', 'store_id']:
            X_train[col] = X_train[col].astype('category').cat.codes
            X_val[col] = X_val[col].astype('category').cat.codes

        # Align columns
        X_val = X_val[X_train.columns]

        model = build_model()

        start = time.time()

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)

        mae = np.mean(np.abs(y_val - preds))

        end = time.time()

        print(f"Fold MAE: {mae:.4f} | Time: {end - start:.2f}s")

        results.append(mae)

    print("\n=========================")
    print("BACKTEST RESULTS")
    print("=========================")

    for i, r in enumerate(results):
        print(f"Fold {i+1}: {r:.4f}")

    print(f"\nMean MAE: {np.mean(results):.4f}")
    print(f"Std MAE: {np.std(results):.4f}")


# =========================
# ENTRY
# =========================
def main():

    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "retail_sales.csv"

    df = pd.read_csv(data_path)

    print("Data loaded:", df.shape)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    run_backtest(df, n_folds=3)


if __name__ == "__main__":
    main()