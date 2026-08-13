import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.features.build_features import build_features_incremental
from src.models.model_wrapper import QuantileDemandModel


def run_forecast_pipeline(df, model, features, horizon=7):
    df = df.copy()

    # =========================
    # 1. Normalize schema
    # =========================
    df = df.rename(columns={
        'item_id': 'product_id',
        'promo': 'promotion'
    })

    df['date'] = pd.to_datetime(df['date'])

    if 'store_id' not in df.columns:
        df['store_id'] = 0

    df = df.sort_values(['store_id', 'product_id', 'date'])

    last_date = df['date'].max()

    # =========================
    # 2. Latest state per SKU
    # =========================
    latest_df = (
        df.groupby(['store_id', 'product_id'])
        .tail(1)[['store_id', 'product_id', 'price', 'promotion']]
        .copy()
    )

    history_df = df.copy()
    results = []

    # =========================
    # 3. Recursive Forecasting
    # =========================
    for step in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=step)

        future_df = latest_df.copy()
        future_df['date'] = future_date
        future_df['sales'] = np.nan

        # Incremental features
        current = build_features_incremental(history_df, future_df)

        X = current.drop(columns=['sales', 'date'])

        # Encode categorical safely
        for col in ['product_id', 'store_id']:
            if col in X.columns:
                X[col] = X[col].astype('category').cat.codes

        # Align columns with training
        X = X.reindex(columns=features)

        # =========================
        # 4. Quantile Prediction
        # =========================
        preds = model.predict(X)

        # Extract quantiles
        p10 = np.clip(preds[0.1], 0, None)
        p50 = np.clip(preds[0.5], 0, None)
        p90 = np.clip(preds[0.9], 0, None)

        # =========================
        # 5. Recursive Update (CRITICAL)
        # =========================
        current['sales'] = p50  # use median for stability

        current['prediction_p10'] = p10
        current['prediction_p50'] = p50
        current['prediction_p90'] = p90

        # Update history for next step
        history_df = pd.concat([history_df, current], ignore_index=True)

        results.append(
            current[
                [
                    'store_id',
                    'product_id',
                    'date',
                    'prediction_p10',
                    'prediction_p50',
                    'prediction_p90'
                ]
            ]
        )

    return pd.concat(results).reset_index(drop=True)


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # =========================
    # Load Quantile Model
    # =========================
    model_path = BASE_DIR / "artifacts" / "lgbm_quantile_models.pkl"
    features_path = BASE_DIR / "artifacts" / "features.pkl"

    model = QuantileDemandModel.load(model_path)
    features = joblib.load(features_path)

    # =========================
    # Load Data
    # =========================
    data_path = BASE_DIR / "data" / "retail_sales.csv"
    df = pd.read_csv(data_path)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # =========================
    # Run Forecast
    # =========================
    forecast_df = run_forecast_pipeline(df, model, features, horizon=7)

    # =========================
    # Save Output
    # =========================
    save_path = BASE_DIR / "data" / "forecast_output.csv"
    forecast_df.to_csv(save_path, index=False)

    print("Forecast saved:", save_path)


if __name__ == "__main__":
    main()