"""
LEGACY MODULE – NOT USED IN PRODUCTION.

This module was part of an earlier prototype using a 19/33-feature schema.
The current production elasticity and optimization logic lives in:
    src/forecast_scenarios.py
using the 50-feature model aligned with artifacts/lgbm_quantile_models.pkl.

Do not modify this file unless intentionally refactoring legacy code.
"""
import numpy as np
import pandas as pd


def _update_price_features(row, new_price):
    row = row.copy()

    # Update core price
    old_price = row["price"]
    row["price"] = new_price

    # Update dependent features (IMPORTANT)
    if "price_lag1" in row:
        row["price_change"] = new_price / row["price_lag1"] - 1

    if "price_avg_30d" in row:
        row["price_vs_avg"] = new_price / row["price_avg_30d"]

    return row


def _predict(model, row):
    if isinstance(row, pd.Series):
        row = row.to_frame().T

    # ensure numeric
    row = row.apply(pd.to_numeric, errors='coerce').fillna(0)

    return model.predict(row)[0]


import numpy as np
import pandas as pd


def estimate_elasticity(
    model,
    row,
    price_col="price",
    pct_change=0.05
):
    """
    Model-based price elasticity estimation.

    Uses feature-consistent perturbation.
    """

    # =========================
    # Base row
    # =========================
    base = row.copy()

    if isinstance(base, pd.Series):
        base = base.to_frame().T

    base = base.copy()

    # =========================
    # Base prediction
    # =========================
    base_pred = model.predict(base)[0]

    # Prevent divide-by-zero
    base_pred = max(base_pred, 1e-6)

    # =========================
    # Perturbed row
    # =========================
    perturbed = base.copy()

    old_price = float(base[price_col].iloc[0])
    new_price = old_price * (1 + pct_change)

    perturbed.loc[:, price_col] = new_price

    # =========================
    # IMPORTANT:
    # Update derived price features
    # =========================

    # price_change
    if "price_lag1" in perturbed.columns:
        lag_price = max(
            float(perturbed["price_lag1"].iloc[0]),
            1e-6
        )

        perturbed.loc[:, "price_change"] = (
            new_price / lag_price - 1
        )

    # price_vs_avg
    if "price_avg_30d" in perturbed.columns:
        avg_price = max(
            float(perturbed["price_avg_30d"].iloc[0]),
            1e-6
        )

        perturbed.loc[:, "price_vs_avg"] = (
            new_price / avg_price
        )

    # =========================
    # New prediction
    # =========================
    new_pred = model.predict(perturbed)[0]
    new_pred = max(new_pred, 1e-6)

    # =========================
    # Elasticity
    # =========================
    pct_demand_change = (
        (new_pred - base_pred) / base_pred
    )

    elasticity = (
        pct_demand_change / pct_change
    )

    return float(elasticity)
