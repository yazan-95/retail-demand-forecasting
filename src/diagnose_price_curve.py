import numpy as np
import pandas as pd

from src.forecast_scenarios_fixed_unit_cost import (
    QuantileDemandModel,
    build_features,
    apply_category_mappings,
    UNIT_COST,
)


# =========================================================
# CONFIGURATION
# =========================================================

PRODUCT_ID = "item_1"
STORE_ID = "store_1"

DATA_PATH = "data/retail_sales.csv"
MODEL_PATH = "artifacts/lgbm_quantile_models.pkl"
MAPPINGS_PATH = "artifacts/category_mappings.pkl"
FEATURES_PATH = "artifacts/features.pkl"


# =========================================================
# PRICE FEATURE UPDATE
# Same logic used by compute_elasticities()
# =========================================================

def update_price_features(row, new_price):

    row = row.copy()

    old_price = float(row["price"].iloc[0])

    row["price"] = new_price

    # Current price changes
    row["price_change_1"] = new_price - old_price

    if "price_lag_1" in row.columns:
        row["price_lag_1"] = old_price

    if "price_lag_7" in row.columns:
        row["price_lag_7"] = old_price

    if "price_lag_28" in row.columns:
        row["price_lag_28"] = old_price

    if "price_avg_7" in row.columns:
        row["price_avg_7"] = (
            row["price_avg_7"] + new_price
        ) / 2

    if "price_avg_28" in row.columns:
        row["price_avg_28"] = (
            row["price_avg_28"] + new_price
        ) / 2

    if "price_std_28" in row.columns:
        row["price_std_28"] = (
            row["price_std_28"] + abs(new_price - old_price)
        ) / 2

    if "price_change_7" in row.columns:
        row["price_change_7"] = new_price - old_price

    if "price_change_28" in row.columns:
        row["price_change_28"] = new_price - old_price

    if "price_vs_avg_7" in row.columns:
        row["price_vs_avg_7"] = (
            new_price / max(float(row["price_avg_7"].iloc[0]), 1e-6)
        )

    if "price_vs_avg_28" in row.columns:
        row["price_vs_avg_28"] = (
            new_price / max(float(row["price_avg_28"].iloc[0]), 1e-6)
        )

    if "price_zscore_28" in row.columns:
        mean_28 = float(row["price_avg_28"].iloc[0])
        std_28 = max(float(row["price_std_28"].iloc[0]), 1e-6)

        row["price_zscore_28"] = (
            new_price - mean_28
        ) / std_28

    if "price_momentum" in row.columns:
        row["price_momentum"] = (
            new_price - old_price
        )

    if "price_volatility" in row.columns:
        row["price_volatility"] = abs(
            new_price - old_price
        )

    if "price_squared" in row.columns:
        row["price_squared"] = new_price ** 2

    if "log_price" in row.columns:
        row["log_price"] = np.log1p(new_price)

    if "discount_depth" in row.columns:
        row["discount_depth"] = 0.0

    return row


# =========================================================
# LOAD ARTIFACTS
# =========================================================

print("=" * 70)
print("PRICE RESPONSE DIAGNOSTIC")
print("=" * 70)

print(f"SKU:        {PRODUCT_ID}")
print(f"Store:      {STORE_ID}")
print(f"UNIT_COST:  {UNIT_COST}")
print()


model = QuantileDemandModel.load(
    MODEL_PATH
)

print("Quantile model loaded.")


category_mappings = pd.read_pickle(
    MAPPINGS_PATH
)

features = pd.read_pickle(
    FEATURES_PATH
)

print("Category mappings loaded.")
print("Features loaded:", len(features))


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

df = df.rename(
    columns={
        "item_id": "product_id",
        "promo": "promotion",
    }
)

df["date"] = pd.to_datetime(df["date"])

print("Data loaded:", df.shape)


# =========================================================
# BUILD PRODUCTION FEATURES
# =========================================================

print("\nBuilding production features...")

feature_df = build_features(df)

feature_df = feature_df[
    (feature_df["product_id"] == PRODUCT_ID)
    & (feature_df["store_id"] == STORE_ID)
].copy()

if feature_df.empty:
    raise ValueError(
        f"No feature rows found for "
        f"{STORE_ID} / {PRODUCT_ID}"
    )

feature_df = feature_df.sort_values("date")

base_row = feature_df.iloc[[-1]].copy()

base_price = float(
    base_row["price"].iloc[0]
)

print(f"Base price: {base_price:.4f}")


# =========================================================
# PRICE GRID
# 70% → 160%
# =========================================================

price_grid = np.linspace(
    base_price * 0.70,
    base_price * 1.60,
    37
)


results = []


# =========================================================
# PRICE RESPONSE LOOP
# =========================================================

for price in price_grid:

    temp_row = update_price_features(
        base_row,
        float(price)
    )

    X = temp_row.drop(
        columns=["sales", "date"],
        errors="ignore"
    )

    X = apply_category_mappings(
        X,
        category_mappings
    )

    X = X.reindex(
        columns=features,
        fill_value=0
    )

    X = (
        X.apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    predictions = model.predict(X)

    demand = float(
        np.clip(
            predictions[0.5],
            0,
            None
        )[0]
    )

    revenue = demand * float(price)

    cost = demand * UNIT_COST

    profit = revenue - cost

    results.append(
        {
            "price": float(price),
            "price_vs_base": float(price / base_price),
            "demand_p50": demand,
            "revenue": revenue,
            "cost": cost,
            "profit_p50": profit,
        }
    )


results_df = pd.DataFrame(results)


# =========================================================
# DISPLAY CURVE
# =========================================================

print("\n" + "=" * 70)
print("PRICE → DEMAND → PROFIT CURVE")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "price": "{:.2f}".format,
            "price_vs_base": "{:.3f}".format,
            "demand_p50": "{:.4f}".format,
            "revenue": "{:.2f}".format,
            "cost": "{:.2f}".format,
            "profit_p50": "{:.2f}".format,
        },
    )
)


# =========================================================
# FIND OPTIMUM
# =========================================================

best_idx = results_df["profit_p50"].idxmax()

best = results_df.loc[best_idx]


print("\n" + "=" * 70)
print("BEST PRICE IN DIAGNOSTIC RANGE")
print("=" * 70)

print(
    f"Base price:       {base_price:.4f}"
)

print(
    f"Best price:       {best['price']:.4f}"
)

print(
    f"Price multiplier: {best['price_vs_base']:.4f}"
)

print(
    f"P50 demand:       {best['demand_p50']:.4f}"
)

print(
    f"P50 revenue:      {best['revenue']:.2f}"
)

print(
    f"Total cost:       {best['cost']:.2f}"
)

print(
    f"P50 profit:       {best['profit_p50']:.2f}"
)


# =========================================================
# +30% CHECK
# =========================================================

plus_30_price = base_price * 1.30

plus_30_idx = (
    np.abs(
        results_df["price"] - plus_30_price
    )
).argmin()

plus_30 = results_df.iloc[plus_30_idx]


print("\n" + "=" * 70)
print("BOUNDARY CHECK")
print("=" * 70)

print(
    f"+30% price:       {plus_30['price']:.4f}"
)

print(
    f"+30% demand:       {plus_30['demand_p50']:.4f}"
)

print(
    f"+30% profit:       {plus_30['profit_p50']:.2f}"
)

print()

if best_idx == len(results_df) - 1:

    print(
        "RESULT: Profit is STILL increasing at +60%."
    )

    print(
        "The diagnostic range is insufficient."
    )

else:

    print(
        "RESULT: Profit peaks INSIDE the diagnostic range."
    )

    print(
        "An interior optimum exists."
    )

print("=" * 70)