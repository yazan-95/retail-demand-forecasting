
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import traceback

from src.features.build_features import (
    build_features,
    build_features_incremental
)
from src.pipelines.train_pipeline import (
    apply_category_mappings,
    normalize_schema
)

from src.models.model_wrapper import QuantileDemandModel

from src.decision.elasticity_engine import estimate_elasticity
from src.decision.optimization_engine import optimize_prices

def predict_p50_demand(
    model,
    row_df
):
    """
    Predict P50 demand using the production quantile wrapper.

    QuantileDemandModel.predict() applies np.expm1() internally,
    converting the model output from log1p demand back to demand units.

    Run with:
        python -m src.forecast_scenarios
    """

    predictions = model.predict(row_df)

    if 0.5 not in predictions:
        raise KeyError(
            "P50 prediction is missing from model output."
        )

    values = np.asarray(
        predictions[0.5],
        dtype=float
    ).reshape(-1)

    if len(values) != 1:
        raise ValueError(
            "predict_p50_demand expects exactly one row, "
            f"received {len(values)} predictions."
        )

    return float(values[0])

UNIT_COST = 15.0  # Fixed unit cost; adjust based on the business economics
ELASTICITY_SHOCK_PCT = 0.05

# =========================
# SCENARIO DEFINITIONS
# =========================
def apply_scenario(df, scenario):
    df = df.copy()

    if scenario == "baseline":
        pass

    elif scenario == "no_promo":
        df['promotion'] = 0

    elif scenario == "full_promo":
        df['promotion'] = 1

    elif scenario == "weekend_promo":
        df['promotion'] = (
            df['date']
            .dt.dayofweek
            .isin([5, 6])
            .astype(int)
        )

    elif scenario == "discount_10":
        df['price'] *= 0.90

    elif scenario == "increase_5":
        df['price'] *= 1.05

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return df


# =========================
# FEATURE IMPORTANCE
# =========================
def print_feature_importance(model):
    print("\n=== FEATURE IMPORTANCE (P50 MODEL) ===")

    importance_df = pd.DataFrame({
        "feature": model.models[0.5].feature_name_,
        "importance": model.models[0.5].feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print(importance_df.head(30))

    # =========================
    # PRICE FEATURE CHECK
    # =========================
    price_features = importance_df[
        importance_df["feature"].str.contains(
            "price",
            case=False,
            na=False
        )
    ]

    print("\n=== PRICE FEATURE IMPORTANCE ===")
    print(price_features.head(20))


# =========================
# FORECAST ENGINE
# =========================
def forecast_scenario(
    df,
    model,
    features,
    scenario,
    horizon=7,
    category_mappings=None
):
    df = df.copy()

    if category_mappings is None:
        raise ValueError(
            "category_mappings are required for inference."
        )

    # =========================
    # Normalize schema
    # =========================
    df = df.rename(columns={
        'item_id': 'product_id',
        'promo': 'promotion'
    })

    df['date'] = pd.to_datetime(df['date'])

    if 'store_id' not in df.columns:
        df['store_id'] = 0

    df = df.sort_values([
        'store_id',
        'product_id',
        'date'
    ])

    last_date = df['date'].max()

    latest_df = (
        df.groupby(['store_id', 'product_id'])
        .tail(1)[[
            'store_id',
            'product_id',
            'price',
            'promotion'
        ]]
        .copy()
    )

    history_df = df.copy()

    results = []

    # =========================
    # RECURSIVE FORECAST
    # =========================
    for step in range(1, horizon + 1):

        future_date = last_date + pd.Timedelta(days=step)

        future_df = latest_df.copy()

        future_df['date'] = future_date
        future_df['sales'] = np.nan

        # Apply scenario
        future_df = apply_scenario(
            future_df,
            scenario
        )

        # Build features
        current = build_features_incremental(
            history_df,
            future_df
        )

        X = current.drop(
            columns=['sales', 'date'],
            errors='ignore'
        )

        # Apply the same deterministic mappings used during training.
        # Keep current unchanged so raw SKU keys remain available for output
        # and recursive history updates.
        X = apply_category_mappings(
            X,
            category_mappings
        )

        # Align features
        X = X.reindex(
            columns=features,
            fill_value=0
        )

        # Force numeric
        X = (
            X.apply(
                pd.to_numeric,
                errors='coerce'
            )
            .fillna(0)
        )

        # =========================
        # QUANTILE PREDICTIONS
        # =========================
        preds = model.predict(X)

        p10 = np.clip(preds[0.1], 0, None)
        p50 = np.clip(preds[0.5], 0, None)
        p90 = np.clip(preds[0.9], 0, None)

        # Enforce quantile monotonicity.
        # Independent quantile models can produce crossing predictions.
        p10, p50, p90 = np.sort(
            np.vstack([p10, p50, p90]),
            axis=0
        )

        # Recursive update
        current['sales'] = p50

        # Predictions
        current['prediction_p10'] = p10
        current['prediction_p50'] = p50
        current['prediction_p90'] = p90

        current['scenario'] = scenario

        # =========================
        # BUSINESS METRICS
        # =========================
        current['revenue_p10'] = (
            p10 * current['price']
        )

        current['revenue_p50'] = (
            p50 * current['price']
        )

        current['revenue_p90'] = (
            p90 * current['price']
        )

        # Fixed unit-cost profit model.
        # Total cost = demand × UNIT_COST.
        current['cost'] = (
            p50 * UNIT_COST
        )

        current['profit_p10'] = (
            p10 * (current['price'] - UNIT_COST)
        )

        current['profit_p50'] = (
            p50 * (current['price'] - UNIT_COST)
        )

        current['profit_p90'] = (
            p90 * (current['price'] - UNIT_COST)
        )

        current['risk_range'] = (
            current['profit_p90']
            - current['profit_p10']
        )

        current['downside'] = (
            current['profit_p10']
        )

        # Recursive history update
        history_df = pd.concat(
            [history_df, current],
            ignore_index=True
        )

        results.append(current)

    return (
        pd.concat(results)
        .reset_index(drop=True)
    )


# =========================
# ELASTICITY + OPTIMIZATION
# =========================
def compute_elasticities(
    df,
    model,
    features,
    max_skus=20,
    verbose=True,
    category_mappings=None
):
    """
    Combined Engine:
    - Real price elasticity estimation
    - Dynamic price optimization
    - Robust LightGBM demand response

    Returns:
    DataFrame:
    - elasticity
    - optimal_price
    - max_profit
    """

    df = df.copy()

    if category_mappings is None:
        raise ValueError(
            "category_mappings are required for inference."
        )

    # =========================
    # NORMALIZE SCHEMA
    # =========================
    df = df.rename(columns={
        'item_id': 'product_id',
        'promo': 'promotion'
    })

    df['date'] = pd.to_datetime(df['date'])

    if 'store_id' not in df.columns:
        df['store_id'] = "store_1"

    df = df.sort_values([
        'store_id',
        'product_id',
        'date'
    ])

    # Build features for the complete historical dataset once.
    #
    # We do not append the latest observation again because it already exists
    # in df. Appending it would change lag and rolling features incorrectly.
    engineered_df = build_features(
        df
    )

    # Select the actual latest engineered row for every SKU.
    latest_features = (
        engineered_df
        .sort_values([
            'store_id',
            'product_id',
            'date'
        ])
        .groupby(
            [
                'store_id',
                'product_id'
            ],
            sort=True
        )
        .tail(1)
        .copy()
    )

    sku_groups = latest_features.groupby(
        [
            'store_id',
            'product_id'
        ],
        sort=True
    )

    results = []

    # =========================
    # LOOP
    # =========================
    for i, ((store_id, product_id), group) in enumerate(sku_groups):

        # PERFORMANCE LIMIT
        if max_skus is not None and i >= max_skus:

            if verbose:
                print(
                    f"\nStopped after "
                    f"{max_skus} SKUs (limit)"
                )

            break

        if verbose and i % 5 == 0:

            print(
                f"Processing SKU {i}: "
                f"{product_id}"
            )

        try:

            # =========================
            # LATEST ENGINEERED OBSERVATION
            # =========================
            #
            # group already contains exactly one correctly selected latest row.
            # Do not call build_features_incremental() here.
            current = group.copy()

            row_df = current.drop(
                columns=['sales', 'date'],
                errors='ignore'
            )

            # Apply persisted training mappings.
            # Do not use pandas cat.codes here because row_df contains
            # only one SKU row at a time.
            row_df = apply_category_mappings(
                row_df,
                category_mappings
            )

            # =========================
            # ALIGN FEATURES
            # =========================
            row_df = row_df.reindex(
                columns=features,
                fill_value=0
            )

            row_df = (
                row_df
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
            )

            # =========================
            # BASE FEATURES
            # =========================
            base_row = row_df.copy()

            base_price = float(
                max(
                    base_row["price"].iloc[0],
                    0.01
                )
            )

            # =========================
            # BASE DEMAND
            # =========================
            base_demand = predict_p50_demand(
                model,
                base_row
            )

            base_demand = max(base_demand, 0.01)

            # =========================
            # PRICE SHOCKS
            # =========================
            price_down = (
                    base_price
                    * (1.0 - ELASTICITY_SHOCK_PCT)
            )
            price_up = (
                    base_price
                    * (1.0 + ELASTICITY_SHOCK_PCT)
            )

            low_row = base_row.copy()
            high_row = base_row.copy()

            # =========================
            # UPDATE PRICE FEATURES
            # =========================
            def update_price_features(temp_row, new_price):

                temp_row = temp_row.copy()
                idx = temp_row.index[0]

                # -------------------------------------------------
                # Normalize numeric columns to float64
                # -------------------------------------------------
                numeric_cols = temp_row.select_dtypes(include=["number"]).columns
                # temp_row[numeric_cols] = temp_row[numeric_cols].astype(float, copy=False)
                temp_row[numeric_cols] = temp_row[numeric_cols].astype(float)

                # Core price
                new_price = float(new_price)
                temp_row.loc[idx, "price"] = new_price

                # ==================================================
                # Price Change Features
                # ==================================================
                if "price_lag_1" in temp_row.columns:
                    lag = max(float(temp_row.loc[idx, "price_lag_1"]), 1e-6)
                    temp_row.loc[idx, "price_change_1"] = np.float32(
                        (new_price - lag) / lag
                    )

                if "price_lag_7" in temp_row.columns:
                    lag = max(float(temp_row.loc[idx, "price_lag_7"]), 1e-6)
                    temp_row.loc[idx, "price_change_7"] = np.float32(
                        (new_price - lag) / lag
                    )

                if "price_lag_28" in temp_row.columns:
                    lag = max(float(temp_row.loc[idx, "price_lag_28"]), 1e-6)
                    temp_row.loc[idx, "price_change_28"] = np.float32(
                        (new_price - lag) / lag
                    )

                # ==================================================
                # Relative Price
                # ==================================================
                if "price_avg_7" in temp_row.columns:
                    avg = max(float(temp_row.loc[idx, "price_avg_7"]), 1e-6)
                    temp_row.loc[idx, "price_vs_avg_7"] = new_price / avg

                if "price_avg_28" in temp_row.columns:
                    avg = max(float(temp_row.loc[idx, "price_avg_28"]), 1e-6)
                    temp_row.loc[idx, "price_vs_avg_28"] = new_price / avg

                # ==================================================
                # Price Z-score
                # ==================================================
                if (
                        "price_avg_28" in temp_row.columns
                        and
                        "price_std_28" in temp_row.columns
                ):
                    avg = float(temp_row.loc[idx, "price_avg_28"])

                    std = max(
                        float(temp_row.loc[idx, "price_std_28"]), 1e-6
                    )

                    temp_row.loc[idx, "price_zscore_28"] = (new_price - avg) / std

                # ==================================================
                # Discount Depth
                # ==================================================
                if "price_vs_avg_28" in temp_row.columns:
                    temp_row.loc[idx, "discount_depth"] = 1.0 - temp_row.loc[idx, "price_vs_avg_28"]

                # ==================================================
                # Price Momentum
                # ==================================================
                if (
                        "price_change_1" in temp_row.columns
                        and
                        "price_change_7" in temp_row.columns
                ):
                    temp_row.loc[idx, "price_momentum"] = (
                            temp_row.loc[idx, "price_change_1"]
                        - temp_row.loc[idx, "price_change_7"]
                    )

                # ==================================================
                # Nonlinear Price Features
                # ==================================================
                if "price_squared" in temp_row.columns:
                    temp_row.loc[idx, "price_squared"] = new_price ** 2

                if "log_price" in temp_row.columns:
                    temp_row.loc[idx, "log_price"] = np.log1p(new_price)

                # ==================================================
                # Promotion Interaction
                # ==================================================
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

            low_row = update_price_features(
                low_row,
                price_down
            )

            high_row = update_price_features(
                high_row,
                price_up
            )

            # =========================
            # DEMAND PREDICTIONS
            # =========================

            # demand_low = float(
            #     model.models[0.5]
            #     .predict(low_row)[0]
            # )

            demand_low = predict_p50_demand(
                model,
                low_row
            )

            demand_high = predict_p50_demand(
                model,
                high_row
            )

            demand_low = max(demand_low, 0.01)
            demand_high = max(demand_high, 0.01)

            # =========================
            # TRUE ELASTICITY
            # =========================
            pct_demand_change = (
                (demand_high - demand_low)
                / ((demand_high + demand_low) / 2)
            )

            pct_price_change = (
                (price_up - price_down)
                / ((price_up + price_down) / 2)
            )

            elasticity = (
                pct_demand_change
                / pct_price_change
            )

            # =========================
            # SANITY FIXES
            # =========================
            if np.isnan(elasticity):
                elasticity = 0.0

            if np.isinf(elasticity):
                elasticity = 0.0

            # Clamp extreme values
            elasticity = float(
                np.clip(
                    elasticity,
                    -10,
                    10
                )
            )

            # =========================
            # OPTIMIZATION ENGINE
            # =========================
            price_grid = np.linspace(
                base_price * 0.70,
                base_price * 1.30,
                25
            )

            best_price = base_price
            best_profit = -np.inf

            for price in price_grid:

                temp_row = base_row.copy()

                temp_row = update_price_features(
                    temp_row,
                    price
                )

                demand = predict_p50_demand(
                    model,
                    temp_row
                )

                demand = max(demand, 0)

                revenue = demand * price
                cost = demand * UNIT_COST
                profit = revenue - cost

                if profit > best_profit:

                    best_profit = profit
                    best_price = price

            # =========================
            # DEBUG OUTPUT
            # =========================
            if verbose:

                print(
                    f"SKU={product_id} | "
                    f"BasePrice={base_price:.2f} | "
                    f"DemandLow={demand_low:.2f} | "
                    f"DemandHigh={demand_high:.2f} | "
                    f"Elasticity={elasticity:.4f}"
                )

        except Exception as e:

            if verbose:
                print(
                    f"Optimization failed for "
                    f"{product_id}: {type(e).__name__}: {e}"
                )

            elasticity = np.nan
            best_price = np.nan
            best_profit = np.nan

        results.append({
            "store_id": store_id,
            "product_id": product_id,
            "elasticity": elasticity,
            "optimal_price": best_price,
            "max_profit": best_profit
        })

    # =========================
    # FINAL DF
    # =========================
    result_df = pd.DataFrame(results)

    if verbose:

        print(
            "\nElasticity + Optimization finished."
        )

        print(
            f"Processed SKUs: "
            f"{len(result_df)}"
        )

    return result_df


# =========================
# MAIN
# =========================
def main():

    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    model_path = (
        BASE_DIR
        / "artifacts"
        / "lgbm_quantile_models.pkl"
    )

    features_path = (
        BASE_DIR
        / "artifacts"
        / "features.pkl"
    )

    mappings_path = (
            BASE_DIR
            / "artifacts"
            / "category_mappings.pkl"
    )

    data_path = (
        BASE_DIR
        / "data"
        / "retail_sales.csv"
    )

    # =========================
    # VALIDATION
    # =========================
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}"
        )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found at: {features_path}"
        )

    if not mappings_path.exists():
        raise FileNotFoundError(
            f"Category mappings not found at: {mappings_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found at: {data_path}"
        )

    # =========================
    # LOAD MODEL
    # =========================
    model = QuantileDemandModel.load(model_path)

    features = joblib.load(
        features_path
    )

    category_mappings = joblib.load(
        mappings_path
    )

    print(
        "Category mappings loaded:",
        {
            key: len(value)
            for key, value in category_mappings.items()
        }
    )

    print("Quantile model loaded.")

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    print_feature_importance(model)

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(
        data_path
    )
    print("Data loaded:", df.shape)
    df = normalize_schema(
        df
    )
    print(
        "Normalized columns:",
        list(df.columns)
    )

    # =========================
    # ELASTICITY + OPTIMIZATION
    # =========================
    elasticity_df = compute_elasticities(
        df,
        model,
        features,
        category_mappings=category_mappings
    )
    # Keep the first corrected run limited to the same SKUs
    # for which elasticity and optimization were calculated.
    decision_keys = (
        elasticity_df[
            [
                "store_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if decision_keys.empty:
        raise ValueError(
            "No SKU decisions were produced."
        )

    scenario_df = df.merge(
        decision_keys,
        on=[
            "store_id",
            "product_id",
        ],
        how="inner"
    )

    scenario_group_count = (
        scenario_df[
            [
                "store_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if scenario_group_count != len(decision_keys):
        raise ValueError(
            "Scenario filtering lost decision SKU keys. "
            f"Expected {len(decision_keys)}, "
            f"got {scenario_group_count}."
        )

    print(
        "\nScenario SKU scope:"
    )

    print(
        f"Decision SKUs: {len(decision_keys)}"
    )

    print(
        f"Scenario rows: {len(scenario_df):,}"
    )

    print("\n=== Elasticity Sample ===")
    print(elasticity_df.head())

    # =========================
    # SCENARIOS
    # =========================
    scenarios = [
        "baseline",
        "no_promo",
        "full_promo",
        "weekend_promo",
        "discount_10",
        "increase_5"
    ]

    all_results = []

    for scenario in scenarios:

        result = forecast_scenario(
            scenario_df,
            model,
            features,
            scenario,
            category_mappings=category_mappings
        )

        print(
            f"{scenario} → "
            f"Profit (P50): "
            f"{result['profit_p50'].sum():.2f}, "
            f"Worst: "
            f"{result['profit_p10'].sum():.2f}, "
            f"Best: "
            f"{result['profit_p90'].sum():.2f}"
        )

        all_results.append(result)

    final_df = pd.concat(all_results)

    # =========================
    # MERGE ELASTICITY
    # =========================
    final_df = final_df.merge(
        elasticity_df,
        on=['store_id', 'product_id'],
        how='left'
    )

    # =========================
    # SUMMARY
    # =========================
    summary = final_df.groupby("scenario")[[
        "profit_p50",
        "profit_p10",
        "profit_p90",
        "risk_range"
    ]].sum()

    print(
        "\n=== Scenario Comparison "
        "(RISK-AWARE) ==="
    )

    print(summary)

    # =========================
    # SAVE
    # =========================
    save_path = (
        BASE_DIR
        / "data"
        / "forecast_scenarios_output.csv"
    )

    final_df.to_csv(
        save_path,
        index=False
    )

    print("\nSaved to:", save_path)


if __name__ == "__main__":
    main()

