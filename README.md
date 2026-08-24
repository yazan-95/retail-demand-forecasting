Questions the system must answer:
• What will demand be in the next 7–30 days?
• Which products are at risk of stockout?
• How much should we reorder?

## Running the system

### 1. Train the production model (one-time or periodic)

From the project root:

```bash
python -m src.pipelines.train_pipeline
```

This produces:

- `artifacts/lgbm_quantile_models.pkl` – 3 quantile LightGBM models (P10, P50, P90).
- `artifacts/features.pkl` – list of 50 feature names in order.
- `artifacts/category_mappings.pkl` – deterministic mappings for `product_id` and `store_id`.

Do not modify these artifacts manually. All inference must use them as-is.

---

### 2. Run scenario forecasting + elasticity + optimization

```bash
python -m src.forecast_scenarios
```

This:

- Loads the production model and mappings.
- Estimates price elasticity and optimal prices for up to 20 SKUs (configurable via `max_skus` in `compute_elasticities()`).
- Runs 6 scenarios: `baseline`, `no_promo`, `full_promo`, `weekend_promo`, `discount_10`, `increase_5`.
- Produces P10/P50/P90 forecasts for demand, revenue, and profit.
- Saves output to `data/forecast_scenarios_output.csv`.

Key outputs:

- `elasticity`, `optimal_price`, `max_profit` per SKU.
- Scenario-level profit summaries (risk-aware: P10/P50/P90).

---

### 3. Run validation tests

Run all checks:

```bash
python -m src.pipelines.run_all_validations
```

Or individually:

```bash
# Core artifact + feature + prediction checks
python -m src.pipelines.validate_production

# Price perturbation sanity check
python -m src.features.test_price_perturbation

# Grouped rolling sanity check
python -m src.features.test_grouped_rolling
```

All tests must pass before deploying model changes or modifying feature engineering.

---

### 4. Business interpretation

- Use `forecast_scenarios_output.csv` to compare scenarios by:
  - `profit_p50` (expected profit),
  - `profit_p10` (conservative / downside),
  - `profit_p90` (upside),
  - `risk_range = profit_p90 - profit_p10`.
- Elasticity values are arc elasticities estimated via a ±5% price shock.
- Optimal prices are chosen by grid search around the base price (±30%, 25 points) maximizing `profit = demand * price - 0.70 * price`.

See `docs/assumptions.md` for full modeling and business assumptions.


## Business sanity checks
See `docs/business_sanity_check_item_1.md` for an example interpretation of
scenario outputs and elasticity for a single SKU (`item_1`).
