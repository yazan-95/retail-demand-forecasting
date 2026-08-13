def compute_feature_stats(df):
    stats = {}

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std()
        }

    return stats