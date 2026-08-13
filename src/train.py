import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor


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

    # Time features
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    # Seasonality (cyclical)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    group = df.groupby(['store_id', 'product_id'])

    # Lags
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = group['sales'].shift(lag)

    base = group['sales'].shift(1)

    # Rolling windows
    for w in [7, 14, 28]:
        df[f'rolling_mean_{w}'] = base.rolling(w, min_periods=3).mean()
        df[f'rolling_std_{w}'] = base.rolling(w, min_periods=3).std()

    # Trend
    df['trend_7_14'] = df['rolling_mean_7'] - df['rolling_mean_14']
    df['trend_7_28'] = df['rolling_mean_7'] - df['rolling_mean_28']

    # Velocity & acceleration
    df['velocity'] = df['lag_1'] - df['lag_7']
    df['acceleration'] = (df['lag_1'] - df['lag_7']) - (df['lag_7'] - df['lag_14'])

    # Price features
    df['price_change'] = group['price'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    df['price_vs_avg'] = df['price'] / (group['price'].shift(1).rolling(7).mean())

    # Promo features
    df['promo_last_week'] = group['promotion'].shift(7).fillna(0)
    df['promo_rolling_7'] = group['promotion'].shift(1).rolling(7).sum()

    # Hierarchical (leakage-safe)
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
# TRAINING PIPELINE
# =========================
def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    data_path = BASE_DIR / "data" / "retail_sales.csv"
    df = pd.read_csv(data_path)

    df = create_features(df)
    df = df.dropna()

    # Time split (IMPORTANT)
    split_date = df['date'].quantile(0.8)

    train_df = df[df['date'] < split_date]
    val_df = df[df['date'] >= split_date]

    X_train = train_df.drop(columns=['sales', 'date'])
    y_train = np.log1p(train_df['sales'])

    X_val = val_df.drop(columns=['sales', 'date'])
    y_val = np.log1p(val_df['sales'])

    # Encode categoricals
    for col in ['product_id', 'store_id']:
        X_train[col] = X_train[col].astype('category').cat.codes
        X_val[col] = X_val[col].astype('category').cat.codes

    # Model (TUNED)
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=255,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
    )

    artifact = {
        "model": model,
        "features": X_train.columns.tolist(),
        "target_transform": "log1p"
    }

    save_path = BASE_DIR / "notebooks" / "lgbm_sales_model_v2.pkl"
    joblib.dump(artifact, save_path)

    print("Model saved to:", save_path)


if __name__ == "__main__":
    main()