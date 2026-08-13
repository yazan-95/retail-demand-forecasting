import joblib
from features import create_features

def load_model(path="models/lgbm_sales_model.pkl"):
    artifact = joblib.load(path)
    return artifact["model"], artifact["features"]

def predict(df):
    model, features = load_model()

    df = create_features(df)
    df = df.dropna()

    X = df[features]

    preds = model.predict(X)

    df['prediction'] = preds

    return df