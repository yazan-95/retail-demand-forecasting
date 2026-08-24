"""
Sanity test for grouped rolling in build_features().

Run with:
    python -m src.features.test_grouped_rolling
"""

import pandas as pd
import numpy as np
from src.features.build_features import build_features


def test_grouped_rolling():
    # 2 stores x 2 products x 5 days
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []

    for store in ["store_1", "store_2"]:
        for product in ["item_1", "item_2"]:
            for i, date in enumerate(dates):
                # store_1/item_1 has increasing sales; others constant
                sales = 10 + i if (store == "store_1" and product == "item_1") else 20
                rows.append({
                    "date": date,
                    "store_id": store,
                    "product_id": product,
                    "sales": sales,
                    "price": 5.0,
                    "promotion": 0,
                    "weekday": date.weekday(),
                    "month": date.month,
                })

    df = pd.DataFrame(rows)
    engineered = build_features(df)

    # Check store_1, item_1 lag_1 manually
    s1i1 = (
        engineered[
            (engineered["store_id"] == "store_1")
            & (engineered["product_id"] == "item_1")
        ]
        .sort_values("date")
    )

    sales_vals = s1i1["sales"].tolist()
    lag1_vals = s1i1["lag_1"].tolist()

    # build_features() fills initial missing lags with 0.
    # Expected lag_1: [0, 10, 11, 12, 13]
    expected_lag1 = [0.0, 10.0, 11.0, 12.0, 13.0]

    assert lag1_vals == expected_lag1, f"lag_1 mismatch: {lag1_vals} vs {expected_lag1}"

    # Check that store_2, item_2 lag_1 does not see store_1, item_1 sales
    s2i2 = (
        engineered[
            (engineered["store_id"] == "store_2")
            & (engineered["product_id"] == "item_2")
        ]
        .sort_values("date")
    )

    lag1_s2i2 = s2i2["lag_1"].fillna(-999).tolist()
    # All sales are constant 20; initial lag is filled with 0 by build_features().
    # Expected lag_1: [0, 20, 20, 20, 20]
    expected_lag1_s2i2 = [0.0, 20.0, 20.0, 20.0, 20.0]

    assert lag1_s2i2 == expected_lag1_s2i2, f"store_2/item_2 lag_1 incorrect: {lag1_s2i2}"

    print("✓ Grouped rolling does not cross store/product boundaries")


if __name__ == "__main__":
    test_grouped_rolling()