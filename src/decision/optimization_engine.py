import numpy as np
import pandas as pd



def generate_price_grid(base_price, steps=15, range_pct=0.3):
    return np.linspace(
        base_price * (1 - range_pct),
        base_price * (1 + range_pct),
        steps
    )


def prepare_features(row_df, features):
    row_df = row_df.copy()

    for col in ['product_id', 'store_id']:
        if col in row_df.columns:
            row_df[col] = row_df[col].astype('category').cat.codes

    row_df = row_df.reindex(columns=features, fill_value=0)
    row_df = row_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return row_df


def update_price_features(row, new_price):
    row = row.copy()

    old_price = row["price"]
    row["price"] = new_price

    if "price_lag1" in row:
        row["price_change"] = new_price / row["price_lag1"] - 1

    if "price_avg_30d" in row:
        row["price_vs_avg"] = new_price / row["price_avg_30d"]

    return row


def predict_demand(model, row_df):
    if isinstance(row_df, pd.Series):
        row_df = row_df.to_frame().T

    return model.predict(row_df)[0]


def optimize_price_for_sku(model, row_df, features, cost_ratio=0.7):
    base_price = row_df["price"].iloc[0]

    price_grid = generate_price_grid(base_price)

    best_price = base_price
    best_profit = -np.inf

    results = []

    for price in price_grid:
        temp_row = row_df.copy()
        temp_row.iloc[0] = update_price_features(temp_row.iloc[0], price)

        temp_row = prepare_features(temp_row, features)

        demand = predict_demand(model, temp_row)

        demand = max(demand, 0)

        revenue = demand * price
        cost = price * cost_ratio
        profit = revenue - cost

        results.append((price, demand, profit))

        if profit > best_profit:
            best_profit = profit
            best_price = price

    return {
        "optimal_price": float(best_price),
        "max_profit": float(best_profit),
        "price_curve": results
    }


def optimize_prices(df, model, features, max_skus=20, verbose=True):
    df = df.copy()

    df = df.rename(columns={
        'item_id': 'product_id',
        'promo': 'promotion'
    })

    df['date'] = pd.to_datetime(df['date'])

    if 'store_id' not in df.columns:
        df['store_id'] = 0

    df = df.sort_values(['store_id', 'product_id', 'date'])

    results = []

    for i, ((store_id, product_id), group) in enumerate(
        df.groupby(['store_id', 'product_id'])
    ):

        if max_skus and i >= max_skus:
            if verbose:
                print(f"\nStopped after {max_skus} SKUs (optimization limit)")
            break

        if verbose and i % 5 == 0:
            print(f"Optimizing SKU {i}: {product_id}")

        latest = group.tail(1)

        current = latest.copy()
        current['sales'] = np.nan

        # Build features ONCE
        from src.features.build_features import build_features_incremental
        current = build_features_incremental(df, current)

        row_df = current.drop(columns=['sales', 'date'], errors='ignore')

        # try:
        #     result = optimize_price_for_sku(
        #         model.models[0.5],
        #         row_df,
        #         features
        #     )
        #
        #     results.append({
        #         "store_id": store_id,
        #         "product_id": product_id,
        #         "optimal_price": result["optimal_price"],
        #         "max_profit": result["max_profit"]
        #     })
        #
        # except Exception as e:
        #     if verbose:
        #         print(f"Optimization error for {product_id}: {e}")

        import traceback
        try:
            result = optimize_price_for_sku(
                model.models[0.5],
                row_df,
                features
                    )

            results.append({
                "store_id": store_id,
                "product_id": product_id,
                "optimal_price": result["optimal_price"],
                "max_profit": result["max_profit"]
                    })
        except Exception:
            traceback.print_exc()
            raise

        return pd.DataFrame(results)