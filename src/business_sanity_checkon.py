"""
Business sanity check for a single SKU.

Usage:
    python business_sanity_checkon.py

Change PRODUCT_ID to inspect a different SKU.

Run with:
    python -m src.business_sanity_checkon
"""

import pandas as pd
from pathlib import Path


def main():
    # =========================
    # CONFIG
    # =========================
    PRODUCT_ID = "item_1"  # <-- change this to inspect another SKU

    # =========================
    # LOAD DATA
    # =========================
    # Project root is one level above src/
    BASE_DIR = Path(__file__).resolve().parents[1]
    output_path = BASE_DIR / "data" / "forecast_scenarios_output.csv"

    if not output_path.exists():
        raise FileNotFoundError(
            f"Scenario output not found at:\n{output_path}\n"
            "Run: python -m src.forecast_scenarios"
        )

    df = pd.read_csv(output_path)

    if PRODUCT_ID not in df["product_id"].values:
        raise ValueError(
            f"Product '{PRODUCT_ID}' not found in output. "
            f"Available products: {sorted(df['product_id'].unique())}"
        )

    # =========================
    # FILTER TO ONE SKU
    # =========================
    sku_df = df[df["product_id"] == PRODUCT_ID].copy()

    if sku_df.empty:
        raise ValueError(f"No rows found for product '{PRODUCT_ID}'")

    # =========================
    # ELASTICITY + OPTIMAL PRICE
    # =========================
    meta = sku_df[["product_id", "elasticity", "optimal_price", "max_profit"]].drop_duplicates()

    print("\n" + "=" * 60)
    print(f"BUSINESS SANITY CHECK — SKU: {PRODUCT_ID}")
    print("=" * 60)

    print("\n1) ELASTICITY + OPTIMAL PRICE")
    print(
        f"   Elasticity:     {meta['elasticity'].iloc[0]:.3f}"
    )
    print(
        f"   Optimal price:  {meta['optimal_price'].iloc[0]:.2f}"
    )
    print(
        f"   Max profit:     {meta['max_profit'].iloc[0]:.2f}"
    )

    # =========================
    # PROFIT BY SCENARIO
    # =========================
    profit_by_scenario = (
        sku_df
        .groupby("scenario")[["profit_p10", "profit_p50", "profit_p90"]]
        .sum()
        .reset_index()
    )

    print("\n2) PROFIT BY SCENARIO (SUM OVER FORECAST HORIZON)")
    print(profit_by_scenario.to_string(index=False))

    # =========================
    # DEMAND BY SCENARIO
    # =========================
    demand_by_scenario = (
        sku_df
        .groupby("scenario")[["prediction_p10", "prediction_p50", "prediction_p90"]]
        .sum()
        .reset_index()
    )

    print("\n3) DEMAND BY SCENARIO (SUM OVER FORECAST HORIZON)")
    print(demand_by_scenario.to_string(index=False))

    # =========================
    # PRICE BY SCENARIO
    # =========================
    price_by_scenario = (
        sku_df
        .groupby("scenario")[["price"]]
        .mean()
        .reset_index()
    )

    print("\n4) AVERAGE PRICE BY SCENARIO")
    print(price_by_scenario.to_string(index=False))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
