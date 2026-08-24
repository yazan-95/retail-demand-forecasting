"""
Production validation checks for the retail demand forecasting system.

Run with:
    python -m src.pipelines.validate_production
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.model_wrapper import QuantileDemandModel
from src.pipelines.train_pipeline import apply_category_mappings, normalize_schema
from src.features.build_features import build_features


def main():
    BASE_DIR = Path(__file__).resolve().parents[2]

    model_path = BASE_DIR / "artifacts" / "lgbm_quantile_models.pkl"
    features_path = BASE_DIR / "artifacts" / "features.pkl"
    mappings_path = BASE_DIR / "artifacts" / "category_mappings.pkl"
    data_path = BASE_DIR / "data" / "retail_sales.csv"

    # Load artifacts
    model = QuantileDemandModel.load(model_path)
    features = joblib.load(features_path)
    category_mappings = joblib.load(mappings_path)

    print("=== ARTIFACT VALIDATION ===")

    # 1. Feature count
    assert len(features) == 50, f"Expected 50 features, got {len(features)}"
    print("✓ Feature count = 50")

    # 2. Model quantiles
    assert set(model.quantiles) == {0.1, 0.5, 0.9}
    print("✓ Quantiles = {0.1, 0.5, 0.9}")

    # 3. Category mappings size
    assert len(category_mappings["product_id"]) == 50
    assert len(category_mappings["store_id"]) == 50
    print("✓ Category mappings: 50 products, 50 stores")

    print("\n=== FEATURE ENGINEERING VALIDATION ===")

    # Load a small subset for speed
    df = pd.read_csv(data_path, nrows=200_000)
    df = normalize_schema(df)
    df = df.sort_values(["store_id", "product_id", "date"])

    engineered = build_features(df)

    # 4. Feature columns match
    missing = [f for f in features if f not in engineered.columns]
    assert not missing, f"Missing features in engineered DataFrame: {missing}"

    # Select exactly the 50 features in the correct order (still with raw IDs)
    X_raw = engineered[features].copy()

    # Confirm column order matches
    assert list(X_raw.columns) == features, "Feature column order does not match artifacts/features.pkl"
    print("✓ Feature columns match artifacts/features.pkl (names and order)")

    # Apply category mappings to get numeric store_id / product_id
    X = apply_category_mappings(X_raw, category_mappings)

    # Ensure store_id and product_id are numeric
    assert pd.api.types.is_integer_dtype(X["store_id"]), "store_id is not integer after mapping"
    assert pd.api.types.is_integer_dtype(X["product_id"]), "product_id is not integer after mapping"

    # 5. No NaN in features
    assert X.isna().sum().sum() == 0, "NaN values found in features"
    print("✓ No NaN in feature matrix")

    # 6. No inf in features
    numeric_cols = X.select_dtypes(include=["number"]).columns
    assert np.isfinite(X[numeric_cols].to_numpy(dtype=float)).all(), "Inf values found in features"
    print("✓ No inf in feature matrix")

    print("\n=== PREDICTION VALIDATION ===")

    # Take a single row (already mapped and numeric)
    row = X.iloc[[0]].copy()
    row = row.reindex(columns=features, fill_value=0)

    preds = model.predict(row)

    p10 = preds[0.1][0]
    p50 = preds[0.5][0]
    p90 = preds[0.9][0]

    # 7. P10 <= P50 <= P90
    assert p10 <= p50 <= p90, f"Quantile order violated: {p10}, {p50}, {p90}"
    print(f"✓ P10 <= P50 <= P90: {p10:.2f} <= {p50:.2f} <= {p90:.2f}")

    # 8. Non-negative predictions
    assert p10 >= 0 and p50 >= 0 and p90 >= 0
    print("✓ All quantile predictions non-negative")

    print("\n=== ALL VALIDATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()