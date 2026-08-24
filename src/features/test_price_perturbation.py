"""
Sanity test: price perturbation must not modify historical price features.

Run with:
    python -m src.features.test_price_perturbation
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.pipelines.train_pipeline import normalize_schema
from src.features.build_features import build_features


def update_price_features(temp_row, new_price):
    """
    Update price-dependent features for a hypothetical new price.

    Rules:
    - Only contemporaneous / derived-from-current-price features are changed.
    - Historical price features (lags, rolling stats) are left unchanged.
    - All numeric columns are normalized to float64 to avoid dtype conflicts.
    """

    temp_row = temp_row.copy()
    idx = temp_row.index[0]

    # -------------------------------------------------
    # Normalize numeric columns to float64
    # -------------------------------------------------
    numeric_cols = temp_row.select_dtypes(include=["number"]).columns
    temp_row[numeric_cols] = temp_row[numeric_cols].astype(float)

    # Core price
    new_price = float(new_price)
    temp_row.loc[idx, "price"] = new_price

    # -------------------------------------------------
    # Price Change Features
    # -------------------------------------------------
    if "price_lag_1" in temp_row.columns:
        lag = max(float(temp_row.loc[idx, "price_lag_1"]), 1e-6)
        temp_row.loc[idx, "price_change_1"] = (new_price - lag) / lag

    if "price_lag_7" in temp_row.columns:
        lag = max(float(temp_row.loc[idx, "price_lag_7"]), 1e-6)
        temp_row.loc[idx, "price_change_7"] = (new_price - lag) / lag

    if "price_lag_28" in temp_row.columns:
        lag = max(float(temp_row.loc[idx, "price_lag_28"]), 1e-6)
        temp_row.loc[idx, "price_change_28"] = (new_price - lag) / lag

    # -------------------------------------------------
    # Relative Price
    # -------------------------------------------------
    if "price_avg_7" in temp_row.columns:
        avg = max(float(temp_row.loc[idx, "price_avg_7"]), 1e-6)
        temp_row.loc[idx, "price_vs_avg_7"] = new_price / avg

    if "price_avg_28" in temp_row.columns:
        avg = max(float(temp_row.loc[idx, "price_avg_28"]), 1e-6)
        temp_row.loc[idx, "price_vs_avg_28"] = new_price / avg

    # -------------------------------------------------
    # Price Z-score
    # -------------------------------------------------
    if (
        "price_avg_28" in temp_row.columns
        and "price_std_28" in temp_row.columns
    ):
        avg = float(temp_row.loc[idx, "price_avg_28"])
        std = max(float(temp_row.loc[idx, "price_std_28"]), 1e-6)
        temp_row.loc[idx, "price_zscore_28"] = (new_price - avg) / std

    # -------------------------------------------------
    # Discount Depth
    # -------------------------------------------------
    if "price_vs_avg_28" in temp_row.columns:
        temp_row.loc[idx, "discount_depth"] = 1.0 - temp_row.loc[idx, "price_vs_avg_28"]

    # -------------------------------------------------
    # Price Momentum
    # -------------------------------------------------
    if (
        "price_change_1" in temp_row.columns
        and "price_change_7" in temp_row.columns
    ):
        temp_row.loc[idx, "price_momentum"] = (
            temp_row.loc[idx, "price_change_1"]
            - temp_row.loc[idx, "price_change_7"]
        )

    # -------------------------------------------------
    # Nonlinear Price Features
    # -------------------------------------------------
    if "price_squared" in temp_row.columns:
        temp_row.loc[idx, "price_squared"] = new_price ** 2

    if "log_price" in temp_row.columns:
        temp_row.loc[idx, "log_price"] = np.log1p(new_price)

    # -------------------------------------------------
    # Promotion Interaction
    # -------------------------------------------------
    if (
        "promotion" in temp_row.columns
        and "discount_depth" in temp_row.columns
        and "promo_price_interaction" in temp_row.columns
    ):
        temp_row.loc[idx, "promo_price_interaction"] = (
            temp_row.loc[idx, "promotion"]
            * temp_row.loc[idx, "discount_depth"]
        )

    return temp_row


def test_price_perturbation():
    BASE_DIR = Path(__file__).resolve().parents[2]
    data_path = BASE_DIR / "data" / "retail_sales.csv"

    # Small subset for speed
    df = pd.read_csv(data_path, nrows=50_000)
    df = normalize_schema(df)
    df = df.sort_values(["store_id", "product_id", "date"])

    engineered = build_features(df)

    # Get latest row per SKU
    latest = (
        engineered
        .sort_values(["store_id", "product_id", "date"])
        .groupby(["store_id", "product_id"], sort=True)
        .tail(1)
        .copy()
    )

    row = latest.iloc[[0]].drop(columns=["sales", "date"], errors="ignore")

    original_price = float(row["price"].iloc[0])
    original_lag1 = float(row["price_lag_1"].iloc[0])
    original_avg7 = float(row["price_avg_7"].iloc[0])
    original_lag28 = float(row["price_lag_28"].iloc[0])
    original_avg28 = float(row["price_avg_28"].iloc[0])

    # Perturb price by +10%
    perturbed = update_price_features(row, original_price * 1.10)

    new_price = float(perturbed["price"].iloc[0])
    new_lag1 = float(perturbed["price_lag_1"].iloc[0])
    new_avg7 = float(perturbed["price_avg_7"].iloc[0])
    new_lag28 = float(perturbed["price_lag_28"].iloc[0])
    new_avg28 = float(perturbed["price_avg_28"].iloc[0])

    # Check new price is updated
    assert abs(new_price - original_price * 1.10) < 1e-6, "New price not updated correctly"

    # Historical features must remain unchanged
    assert new_lag1 == original_lag1, f"price_lag_1 changed: {original_lag1} -> {new_lag1}"
    assert new_avg7 == original_avg7, f"price_avg_7 changed: {original_avg7} -> {new_avg7}"
    assert new_lag28 == original_lag28, f"price_lag_28 changed: {original_lag28} -> {new_lag28}"
    assert new_avg28 == original_avg28, f"price_avg_28 changed: {original_avg28} -> {new_avg28}"

    print("✓ Price perturbation does not modify historical price features")


if __name__ == "__main__":
    test_price_perturbation()