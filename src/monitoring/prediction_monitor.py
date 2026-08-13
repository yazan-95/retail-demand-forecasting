def monitor_predictions(preds_df):
    print("\nPrediction Summary:")
    print("Mean:", preds_df["prediction"].mean())
    print("Std:", preds_df["prediction"].std())
    print("Min:", preds_df["prediction"].min())
    print("Max:", preds_df["prediction"].max())