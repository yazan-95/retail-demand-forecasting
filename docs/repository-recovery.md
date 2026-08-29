# Repository Recovery & Integrity Investigation

## Purpose

This document records the recovery and integrity-verification process performed after a destructive repository-history operation affected several important project files.

The objective was to:

1. Preserve the remaining project state.
2. Determine which files and Git objects were still recoverable.
3. Recover original project assets where possible.
4. Verify recovered datasets, models, and production artifacts.
5. Confirm that the restored project remained functionally consistent.
6. Establish safeguards to reduce the risk of similar incidents in the future.

This document is intentionally separate from the main project README.

The `README.md` documents the **retail forecasting and decision-optimization system** itself, while this document documents the **engineering recovery and repository-integrity investigation**.

---

# 1. Incident Summary

During repository maintenance, a history-rewriting Git operation unintentionally removed several important project assets from the repository state.

The affected assets included:

```text
data/retail_sales.csv

artifacts/category_mappings.pkl
artifacts/features.pkl
artifacts/lgbm_quantile_models.pkl

models/lgbm_sales_model.pkl

notebooks/lgbm_sales_model.pkl
notebooks/lgbm_sales_model_v2.pkl
```

The incident was treated as a repository-integrity problem rather than immediately rebuilding the affected assets.

The recovery strategy therefore prioritized:

```text
Preserve
   ↓
Investigate
   ↓
Locate surviving copies / objects
   ↓
Recover
   ↓
Validate
   ↓
Restore project integrity
```

No assumption was made that a file should be regenerated simply because a similarly named file existed elsewhere.

---

# 2. Recovery Principles

The investigation followed several principles.

### 2.1 Preserve evidence

Existing files, Git metadata, recovery copies, and forensic material were preserved before destructive investigation steps were performed.

### 2.2 Prefer recovery over regeneration

Where the original file contents could be recovered, recovery was preferred over retraining or recreating the asset.

This was particularly important for:

* large datasets
* trained machine-learning models
* serialized artifacts
* files whose exact training state might otherwise be impossible to reproduce

### 2.3 Verify before trusting

A file was not considered successfully recovered merely because:

* it existed,
* it had a familiar filename,
* it had a non-zero size, or
* it could be opened as a binary file.

Recovered assets were checked using structural, loading, and project-level validation.

### 2.4 Avoid unnecessary modifications

Once sufficient evidence established that the required project assets were intact, further recovery actions were stopped.

This prevented unnecessary changes to a functioning project.

---

# 3. Affected Project Assets

The investigation initially identified the following important files as affected or potentially affected:

| Asset                                | Purpose                                    |
| ------------------------------------ | ------------------------------------------ |
| `data/retail_sales.csv`              | Original retail training dataset           |
| `artifacts/category_mappings.pkl`    | Production categorical mappings            |
| `artifacts/features.pkl`             | Production feature schema                  |
| `artifacts/lgbm_quantile_models.pkl` | Production P10/P50/P90 models              |
| `models/lgbm_sales_model.pkl`        | Historical/general LightGBM model location |
| `notebooks/lgbm_sales_model.pkl`     | Surviving serialized LightGBM model        |
| `notebooks/lgbm_sales_model_v2.pkl`  | Recovered/available v2 LightGBM model      |

The distinction between **production artifacts**, **historical model files**, and **temporary notebook/checkpoint files** was maintained throughout the investigation.

---

# 4. Evidence Sources

Multiple sources were considered during the investigation.

## 4.1 Git History

Git history and references were inspected to determine whether previous commits still contained usable versions of affected files.

Relevant investigation areas included:

```text
git log
git reflog
git fsck
Git object database
pack files
unreachable objects
```

The investigation confirmed that Git history alone could not be assumed to contain every required version after the history-rewriting operation.

---

## 4.2 Git Object Database

The Git object database was inspected for surviving blobs and historical file contents.

Object-level investigation included:

```text
commits
trees
blobs
pack files
unreachable objects
```

This helped distinguish between:

```text
file no longer referenced
```

and:

```text
file content no longer available in Git
```

That distinction was important before considering filesystem-level recovery.

---

## 4.3 Filesystem and APFS Investigation

The local filesystem was investigated for surviving copies and filesystem-level recovery opportunities.

The investigation included relevant:

```text
project directories
backup directories
APFS snapshots
Trash
local copies
```

No destructive filesystem recovery action was performed merely for the sake of continuing the investigation after sufficient evidence had been obtained.

---

## 4.4 PyCharm Local History

PyCharm Local History was investigated because it can preserve file changes independently of Git.

Relevant Local History data included:

```text
changes.storageData
changes.storageRecordIndex
```

Forensic copies of the relevant Local History data were preserved during the investigation.

This provided an independent recovery source outside the Git object database.

---

# 5. Recovery of the Dataset

The original dataset was successfully restored to:

```text
data/retail_sales.csv
```

The final verified file size was:

```text
195,154,518 bytes
```

The dataset was subsequently validated structurally.

Verified properties:

```text
Rows:       4,565,000
Columns:    8
Stores:     50
Items:      50
Date range: 2019-01-01 → 2023-12-31
Null values: 0
```

The original dataset schema is:

```text
date
store_id
item_id
sales
price
promo
weekday
month
```

The dataset therefore matches the expected project scale:

```text
50 stores × 50 products
= 2,500 store/product groups
```

The dataset was not replaced with a newly generated synthetic dataset.

---

# 6. Recovery of Production Artifacts

The following production artifacts were restored and successfully validated:

```text
artifacts/category_mappings.pkl
artifacts/features.pkl
artifacts/lgbm_quantile_models.pkl
```

Final verified sizes:

```text
category_mappings.pkl
1,280 bytes

features.pkl
723 bytes

lgbm_quantile_models.pkl
56,357,026 bytes
```

The artifacts successfully passed Joblib loading tests.

---

# 7. Production Artifact Validation

The restored artifacts were inspected after loading.

The resulting production state was:

```text
Features:
50

Category mappings:
50 products
50 stores

Quantile models:
0.1
0.5
0.9
```

The production feature schema therefore contains exactly:

```text
50 features
```

and the quantile forecasting layer contains:

```text
P10
P50
P90
```

The restored artifacts were verified to be internally consistent with the production forecasting pipeline.

---

# 8. Model Recovery

Several serialized LightGBM models existed in different project locations.

The investigation therefore distinguished between:

```text
production quantile models
```

and:

```text
historical / notebook model files
```

The following model files were ultimately available and loadable:

```text
notebooks/lgbm_sales_model.pkl
notebooks/lgbm_sales_model_v2.pkl
```

Verified file sizes:

```text
lgbm_sales_model.pkl
19,291,002 bytes

lgbm_sales_model_v2.pkl
28,416,180 bytes
```

Both files successfully passed Joblib loading.

---

# 9. `models/lgbm_sales_model.pkl`

The path:

```text
models/lgbm_sales_model.pkl
```

was found to be an empty file.

This does **not** represent a loss of the usable model required by the current project state.

The functional serialized model exists at:

```text
notebooks/lgbm_sales_model.pkl
```

and successfully loads as a Python dictionary through Joblib.

Therefore:

```text
models/lgbm_sales_model.pkl
```

is treated as an obsolete/empty duplicate path rather than as the authoritative surviving model.

The important engineering distinction is between:

```text
filename/path existence
```

and:

```text
verified usable model artifact
```

The latter was confirmed independently.

---

# 10. `lgbm_sales_model_v2.pkl`

The v2 model was one of the important assets investigated during recovery.

The currently available file is:

```text
notebooks/lgbm_sales_model_v2.pkl
```

Verified size:

```text
28,416,180 bytes
```

It successfully loads through Joblib and is therefore available as a functional serialized model artifact.

The project does not depend on an assumption that this file can be recreated identically through a new training run.

The recovered/available serialized artifact itself is retained.

---

# 11. Checkpoint Files

Several empty checkpoint files were also observed, including names such as:

```text
category_mappings-checkpoint.pkl
features-checkpoint.pkl
lgbm_quantile_models-checkpoint.pkl
retail_sales-checkpoint.csv
lgbm_sales_model-checkpoint.pkl
```

These files are not treated as production project assets.

Checkpoint files can be created by notebook or development tooling and may be incomplete, temporary, or zero-byte files.

The authoritative project files are the production paths documented in:

```text
artifacts/
data/
notebooks/
```

and the corresponding validated source code.

Therefore, empty checkpoint files do not invalidate the recovered project.

---

# 12. Production Scenario Output

The production scenario output was also verified.

File:

```text
data/forecast_scenarios_output.csv
```

Verified state:

```text
Rows: 840
Columns: 68
Scenarios: 6
Decision SKUs: 20
Missing values: 0
```

The expected scenarios were present:

```text
baseline
discount_10
full_promo
increase_5
no_promo
weekend_promo
```

This provided an additional independent indication that the restored production artifacts and scenario engine remained operational.

---

# 13. Scenario and Economic Validation

The restored project was tested beyond simple file existence.

The validation confirmed:

```text
P10 <= P50 <= P90
```

for demand forecasts.

Revenue quantile ordering was also valid.

Profit quantile ordering was examined separately because profit is a transformed business quantity rather than a direct copy of demand quantiles.

This distinction is important.

For a selling price below unit cost:

```text
profit = demand × (price - unit_cost)
```

the profit multiplier becomes negative.

Consequently, higher demand can produce a more negative profit value.

Therefore, observing:

```text
profit_p10 > profit_p50 > profit_p90
```

under a loss-making price does not necessarily indicate a forecasting quantile violation.

The investigation confirmed that the observed 42 profit-ordering cases were associated with:

```text
UNIT_COST = 15.0
price < UNIT_COST
```

for:

```text
item_20
```

and represented economically consistent negative-margin behavior rather than evidence that the demand model itself was corrupted.

---

# 14. Price Optimization Validation

The price optimization layer was independently checked.

The final diagnostic confirmed:

```text
UNIT COST = 15.0
DECISION SKUs = 20
```

Only one current price was below unit cost:

```text
item_20
current price = 12.20
unit margin = -2.80
```

Its estimated optimized price was:

```text
optimal price = 15.86
```

with:

```text
optimal unit margin = 0.86
```

and:

```text
optimal price > unit cost
```

The other decision SKUs had optimal prices above unit cost.

This confirmed that the optimization layer was still producing economically interpretable results.

---

# 15. Read-Only Integrity Audit

A final project-integrity audit was executed without modifying project files.

The audit checked:

```text
required files
dataset structure
Joblib loading
production artifacts
scenario output
source syntax
Git working tree
SHA-256 hashes
```

The final audit confirmed that the required dataset, production artifacts, source files, and usable serialized models were present and readable.

---

# 16. Final Verified Production Assets

The final important assets include:

```text
data/retail_sales.csv

artifacts/category_mappings.pkl
artifacts/features.pkl
artifacts/lgbm_quantile_models.pkl

notebooks/lgbm_sales_model.pkl
notebooks/lgbm_sales_model_v2.pkl

src/forecast_scenarios.py
src/forecast_scenarios_fixed_unit_cost.py

src/decision/elasticity_engine.py
src/decision/optimization_engine.py
```

The production quantile artifacts contain:

```text
50 features
3 quantile models
50 product mappings
50 store mappings
```

---

# 17. Repository State

The final audit showed that the working tree contained expected development and recovery-related modifications.

Examples included:

```text
modified source files
modified production artifacts
modified data outputs
notebook changes
recovered notebook models
recovery backup directories
```

These changes are intentionally preserved until the repository is reviewed and committed deliberately.

The recovery process did not treat:

```text
git status = clean
```

as a prerequisite for declaring the project recovered.

A working tree can contain legitimate development changes while still containing a fully functional project.

---

# 18. Recovery Validation Philosophy

The investigation demonstrated an important distinction between three different levels of verification.

## Level 1 — File verification

Does the file exist?

```text
exists
```

## Level 2 — Artifact verification

Can the file be loaded and structurally inspected?

```text
Joblib load
schema inspection
size verification
```

## Level 3 — System verification

Does the recovered asset work correctly inside the project?

```text
feature validation
prediction validation
scenario validation
economic validation
```

The final recovery relied primarily on Level 3 verification for the important production components.

---

# 19. Engineering Lessons

The incident highlighted several practical software-engineering principles.

### 19.1 Git is not a backup system

Git provides version control, but it should not be treated as the only copy of important datasets or generated model artifacts.

### 19.2 Generated artifacts need lifecycle management

Large serialized models and production datasets should have a defined storage and backup strategy.

### 19.3 Destructive Git operations require explicit caution

History-rewriting commands can affect more than the immediately visible working tree.

They should be performed only after confirming:

```text
backup availability
current branch
current commit
working-tree state
recovery options
```

### 19.4 Recovery should be evidence-driven

Blindly retraining a model can produce a valid model without reproducing the original model.

For recovery purposes, those are not equivalent outcomes.

### 19.5 Validation is part of recovery

A recovered file is not considered trustworthy merely because it opens.

It must be validated against the expectations of the system that consumes it.

---

# 20. Safeguards Going Forward

The project should follow these safeguards for future repository maintenance.

## Before destructive Git operations

Check:

```bash
git status
git branch --show-current
git log --oneline -n 10
```

Create a verified backup of critical assets before rewriting history.

---

## Protect production artifacts

Important generated assets should have an explicit backup strategy:

```text
dataset
production models
feature schema
category mappings
important outputs
```

---

## Verify after repository maintenance

Run the project's validation suite:

```bash
python -m src.pipelines.run_all_validations
```

and perform a read-only integrity audit when major repository changes occur.

---

# 21. Recovery Outcome

The repository recovery was considered successful because the important project components were restored and independently validated.

Final state:

```text
✓ Original-scale retail dataset available
✓ Production quantile artifacts available
✓ 50-feature production schema available
✓ Category mappings available
✓ P10/P50/P90 models loadable
✓ Historical LightGBM model available
✓ lgbm_sales_model_v2.pkl available
✓ Scenario engine operational
✓ Price elasticity operational
✓ Price optimization operational
✓ Business scenario output validated
✓ Source files pass syntax validation
✓ Project-level integrity audit completed
```

The empty:

```text
models/lgbm_sales_model.pkl
```

and temporary checkpoint files do not represent missing production functionality because the corresponding usable assets were independently recovered and validated in their authoritative locations.

---

# 22. Final Engineering Assessment

The recovery process is considered complete.

The important outcome is not simply that individual files were restored.

The more significant outcome is that the project was taken through a complete engineering recovery cycle:

```text
Repository incident
        ↓
Evidence preservation
        ↓
Git investigation
        ↓
Filesystem investigation
        ↓
Local History investigation
        ↓
Asset recovery
        ↓
Artifact loading
        ↓
Dataset validation
        ↓
Model validation
        ↓
Scenario validation
        ↓
Economic validation
        ↓
Final integrity audit
```

This demonstrates a broader engineering capability than model development alone:

**diagnosis, evidence preservation, controlled recovery, validation, and risk-aware decision making.**

The recovered project can therefore continue to be treated as the finalized **Retail Demand Forecasting & Decision Optimization System**, with this document retained as its technical repository-recovery record.
