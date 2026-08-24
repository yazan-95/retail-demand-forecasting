# Business Sanity Check — item_1

**Date:** 2026-08-17  
**Model:** 50-feature LightGBM quantile model (P10/P50/P90)  
**Profit assumption (current):** unit cost = 70% of selling price.

## 1. Elasticity and optimal price

- **Elasticity:** −2.27 (elastic demand)
- **Base price:** 21.30
- **Optimal price (model):** 27.69
- **Max profit (optimization):** 960.54

Interpretation:

- A 1% price increase → ~2.27% demand decrease.
- A 1% price decrease → ~2.27% demand increase.
- The optimizer suggests a ~30% price increase to maximize profit under the current profit model.

## 2. Profit by scenario (P50, sum over 7-day horizon)

| Scenario       | Profit P10 | Profit P50 | Profit P90 |
|---------------|-----------:|-----------:|-----------:|
| baseline      | 6,872.31   | 7,434.54   | 8,208.24   |
| discount_10   | 7,014.54   | 7,545.94   | 8,764.24   |
| full_promo    | 8,615.11   | 9,559.70   | 9,713.43   |
| increase_5    | 6,140.00   | 6,738.26   | 7,660.46   |
| no_promo      | 6,872.31   | 7,434.54   | 8,208.24   |
| weekend_promo | 7,374.40   | 7,875.75   | 8,643.35   |

Key points:

- **`full_promo` is best**:
  - Profit P50: 9,560 vs 7,435 baseline → **+28.6%**.
  - Higher demand (453.7 vs 353.9) at the same price (21.30).
  - Strong positive promotion effect learned by the model.

- **`discount_10`**:
  - Price: 19.17 (−10%).
  - Demand: 398.5 (+12.6% vs baseline).
  - Profit: 7,546 (+1.5% vs baseline).
  - Small profit gain despite lower price.

- **`increase_5`**:
  - Price: 22.365 (+5%).
  - Demand: 306.2 (−13.5% vs baseline).
  - Profit: 6,738 (−9.4% vs baseline).
  - Confirms elastic behavior.

- **`weekend_promo`**:
  - Profit P50: 7,876 (+5.9% vs baseline).
  - Demand: 374.7 (+5.9% vs baseline).

## 3. Demand by scenario (sum over horizon)

| Scenario       | Demand P10 | Demand P50 | Demand P90 |
|---------------|-----------:|-----------:|-----------:|
| baseline      | 327.54     | 353.94     | 390.26     |
| discount_10   | 370.81     | 398.53     | 462.09     |
| full_promo    | 409.37     | 453.71     | 460.93     |
| increase_5    | 279.44     | 306.19     | 347.42     |
| no_promo      | 327.54     | 353.94     | 390.26     |
| weekend_promo | 351.12     | 374.65     | 410.69     |

## 4. Average price by scenario

| Scenario       | Price |
|---------------|------:|
| baseline      | 21.30 |
| discount_10   | 19.17 |
| full_promo    | 21.30 |
| increase_5    | 22.37 |
| no_promo      | 21.30 |
| weekend_promo | 21.30 |

## 5. Business conclusions (current profit model)

- Promotions are highly valuable for `item_1` under the current profit assumption.
- The optimizer’s higher optimal price (27.69) reflects the specific profit formula (`cost = 0.70 * price`), not necessarily a real-world recommendation.
- For practical pricing, consider:
  - Using a **fixed unit cost** instead of 70% of price.
  - Re-running optimization and comparing optimal prices and scenario profits.

## 6. Next steps

- Refine profit model to use a fixed unit cost (e.g. 15.0 or actual cost).
- Re-run scenarios and re-evaluate:
  - Optimal price.
  - Scenario profit rankings.
- Document updated assumptions in `docs/assumptions.md`.