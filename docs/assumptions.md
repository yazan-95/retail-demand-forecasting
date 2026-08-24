# Business and Modeling Assumptions

## Profit model

Unit cost is assumed to be 70% of the selling price.

For a given price \(p\) and predicted demand \(d\):

\[
\text{revenue} = d \cdot p,\quad
\text{cost} = 0.70 \cdot p,\quad
\text{profit} = \text{revenue} - \text{cost}.
\]

This is a simplified margin model; it does not assume a fixed per-unit cost independent of price.

## Target transformation

- The LightGBM quantile models were trained on \(\log(1 + \text{sales})\).
- `QuantileDemandModel.predict()` applies \(\exp(\cdot) - 1\) to return demand units.
- All business metrics (revenue, profit, elasticity, optimization) use demand units, not log-demand.

## Feature engineering

- 50 production features defined in `artifacts/features.pkl`.
- Historical demand features use `shift(1)` to avoid target leakage.
- Rolling and lag features are grouped by `(store_id, product_id)` and do not cross SKU boundaries.
- Initial missing lag/rolling values are filled with 0.

## Category mappings

- `product_id` and `store_id` are mapped to integers using persisted mappings in `artifacts/category_mappings.pkl`.
- These mappings are deterministic and must be reused for all inference.
- Pandas `astype('category').cat.codes` is not used in production inference.

## Quantile forecasts

- Models: P10, P50, P90 quantiles of demand.
- Recursive multi-step forecasting uses P50 to update history.
- Business metrics are computed for all three quantiles to support risk-aware decisions.