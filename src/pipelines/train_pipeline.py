from src.features.build_features import build_features

import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================
QUANTILES = [0.1, 0.5, 0.9]

CATEGORICAL_COLUMNS = [
    "product_id",
    "store_id"
]

MAPPING_FILENAME = "category_mappings.pkl"
MODEL_FILENAME = "lgbm_quantile_models.pkl"
FEATURES_FILENAME = "features.pkl"


# ============================================================
# SCHEMA NORMALIZATION
# ============================================================

def normalize_schema(df):
    """
    Normalize raw retail data into the canonical project schema.

    Canonical columns:
        product_id
        promotion
        store_id
        date
        sales
        price
    """

    df = df.copy()

    df = df.rename(columns={
        "item_id": "product_id",
        "promo": "promotion"
    })

    if "store_id" not in df.columns:
        df["store_id"] = 0

    if "promotion" not in df.columns:
        df["promotion"] = 0

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date"]
    )

    return df


# ============================================================
# DETERMINISTIC CATEGORY MAPPINGS
# ============================================================

def build_category_mappings(df):
    """
    Build deterministic categorical mappings.

    IMPORTANT:
    We intentionally do NOT use pandas .cat.codes.

    The mapping is explicitly created and persisted so that
    training and inference use exactly the same representation.

    Example:

        item_1 -> 0
        item_2 -> 1
        item_3 -> 2

    Unknown categories during inference are mapped to -1.
    """

    mappings = {}

    for col in CATEGORICAL_COLUMNS:

        if col not in df.columns:
            continue

        values = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # Deterministic ordering
        values = sorted(values)

        mapping = {
            value: index
            for index, value in enumerate(values)
        }

        mappings[col] = mapping

        print(
            f"Category mapping created for "
            f"{col}: {len(mapping)} categories"
        )

    return mappings


# ============================================================
# APPLY CATEGORY MAPPINGS
# ============================================================

def apply_category_mappings(
    df,
    mappings,
    unknown_value=-1
):
    """
    Apply persisted categorical mappings.

    Unknown categories are mapped to -1.

    This function must be used for both training and inference.
    """

    df = df.copy()

    for col in CATEGORICAL_COLUMNS:

        if col not in df.columns:
            continue

        mapping = mappings.get(col, {})

        df[col] = (
            df[col]
            .astype(str)
            .map(mapping)
            .fillna(unknown_value)
            .astype(np.int32)
        )

    return df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_training_features(
    df,
    features,
    mappings
):
    """
    Build and prepare the training feature matrix.
    """

    # Feature engineering
    df = build_features(df)

    # Remove invalid target rows
    df = df.dropna(
        subset=["sales"]
    )

    # Apply deterministic categorical mapping
    df = apply_category_mappings(
        df,
        mappings
    )

    # Ensure all expected features exist
    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing training features:\n"
            f"{missing_features}"
        )

    X = df[features].copy()

    y = df["sales"].copy()

    # Numeric safety
    X = (
        X
        .apply(pd.to_numeric, errors="coerce")
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # Explicit float32 representation
    # reduces memory usage for the ~4.5M row dataset.
    X = X.astype(np.float32)

    # Demand must be non-negative
    y = (
        pd.to_numeric(
            y,
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    return X, y, df


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================

def chronological_split(
    df,
    validation_fraction=0.10
):
    """
    Split data using dates rather than row position.

    This is important because the dataset is grouped by
    store/product and therefore a simple 90/10 row split
    does not necessarily represent future forecasting.
    """

    if "date" not in df.columns:
        raise ValueError(
            "Date column is required for "
            "chronological validation."
        )

    unique_dates = (
        pd.Series(
            df["date"]
            .dropna()
            .unique()
        )
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "Not enough unique dates for "
            "chronological validation."
        )

    split_position = int(
        len(unique_dates)
        * (1.0 - validation_fraction)
    )

    split_position = max(
        1,
        min(
            split_position,
            len(unique_dates) - 1
        )
    )

    split_date = unique_dates.iloc[
        split_position
    ]

    train_mask = (
        df["date"] < split_date
    )

    valid_mask = (
        df["date"] >= split_date
    )

    return train_mask, valid_mask, split_date


# ============================================================
# MODEL TRAINING
# ============================================================

def train_quantile_models(
    X_train,
    y_train,
    X_valid,
    y_valid
):
    """
    Train p10, p50 and p90 LightGBM quantile models.
    """

    models = {}

    for q in QUANTILES:

        print(
            f"\n========================================"
        )

        print(
            f"Training quantile model: P{int(q * 100)}"
        )

        print(
            f"========================================"
        )

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,

            n_estimators=1000,
            learning_rate=0.05,

            num_leaves=255,

            subsample=0.8,
            colsample_bytree=0.8,

            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_valid, y_valid)
            ],
            eval_metric="quantile",
            callbacks=[
                lgb.early_stopping(
                    50
                ),
                lgb.log_evaluation(
                    100
                )
            ]
        )

        models[q] = model

        print(
            f"P{int(q * 100)} "
            f"training completed."
        )

    return models


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    models,
    features,
    category_mappings,
    save_dir
):
    """
    Persist all artifacts required by inference.
    """

    save_dir = Path(
        save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        save_dir
        / MODEL_FILENAME
    )

    features_path = (
        save_dir
        / FEATURES_FILENAME
    )

    mappings_path = (
        save_dir
        / MAPPING_FILENAME
    )

    joblib.dump(
        models,
        model_path
    )

    joblib.dump(
        features,
        features_path
    )

    joblib.dump(
        category_mappings,
        mappings_path
    )

    print(
        "\n========================================"
    )

    print(
        "Artifacts saved successfully:"
    )

    print(
        f"Models:    {model_path}"
    )

    print(
        f"Features:  {features_path}"
    )

    print(
        f"Mappings:  {mappings_path}"
    )

    print(
        "========================================"
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_train_pipeline(
    df,
    features,
    save_dir
):
    """
    Production training pipeline.

    Flow:

        raw data
            ↓
        normalize schema
            ↓
        deterministic category mappings
            ↓
        feature engineering
            ↓
        feature preparation
            ↓
        chronological split
            ↓
        p10 / p50 / p90 training
            ↓
        artifact persistence
    """

    print(
        "\n========================================"
    )

    print(
        "STARTING TRAINING PIPELINE"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # 1. Normalize schema
    # --------------------------------------------------------

    df = normalize_schema(
        df
    )

    df = df.sort_values(
        [
            "store_id",
            "product_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Normalized dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 2. Build deterministic mappings
    # --------------------------------------------------------

    category_mappings = (
        build_category_mappings(
            df
        )
    )

    # --------------------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------------------

    print(
        "\nBuilding training features..."
    )

    X, y, engineered_df = (
        prepare_training_features(
            df,
            features,
            category_mappings
        )
    )

    print(
        f"Feature matrix shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    # --------------------------------------------------------
    # 4. Chronological split
    # --------------------------------------------------------

    train_mask, valid_mask, split_date = (
        chronological_split(
            engineered_df,
            validation_fraction=0.10
        )
    )

    X_train = X.loc[
        train_mask
    ]

    X_valid = X.loc[
        valid_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    y_valid = y.loc[
        valid_mask
    ]

    print(
        "\nChronological validation split:"
    )

    print(
        f"Validation start date: {split_date}"
    )

    print(
        f"Training rows:   {len(X_train):,}"
    )

    print(
        f"Validation rows: {len(X_valid):,}"
    )

    # --------------------------------------------------------
    # 5. Log target transformation
    # --------------------------------------------------------

    y_train_log = np.log1p(
        y_train
    )

    y_valid_log = np.log1p(
        y_valid
    )

    # --------------------------------------------------------
    # 6. Train quantile models
    # --------------------------------------------------------

    models = train_quantile_models(
        X_train,
        y_train_log,
        X_valid,
        y_valid_log
    )

    # --------------------------------------------------------
    # 7. Save artifacts
    # --------------------------------------------------------

    save_artifacts(
        models,
        features,
        category_mappings,
        save_dir
    )

    print(
        "\nTraining pipeline completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    data_path = (
        BASE_DIR
        / "data"
        / "retail_sales.csv"
    )

    save_dir = (
        BASE_DIR
        / "artifacts"
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not data_path.exists():

        raise FileNotFoundError(
            f"Dataset not found at:\n"
            f"{data_path}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        f"\nLoading dataset:\n"
        f"{data_path}"
    )

    df = pd.read_csv(
        data_path
    )

    print(
        f"Raw dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Normalize before feature discovery
    # --------------------------------------------------------

    df_tmp = normalize_schema(
        df
    )

    df_tmp = df_tmp.sort_values(
        [
            "store_id",
            "product_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Generate features automatically
    # --------------------------------------------------------

    print(
        "\nGenerating features "
        "for feature discovery..."
    )

    df_tmp = build_features(
        df_tmp
    )

    # --------------------------------------------------------
    # Detect model features
    # --------------------------------------------------------

    excluded_cols = [
        "sales",
        "date"
    ]

    features = [
        col
        for col in df_tmp.columns
        if col not in excluded_cols
    ]

    if not features:

        raise ValueError(
            "No model features were detected."
        )

    print(
        f"\nDetected {len(features)} features:"
    )

    print(
        features
    )

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    run_train_pipeline(
        df,
        features,
        save_dir
    )