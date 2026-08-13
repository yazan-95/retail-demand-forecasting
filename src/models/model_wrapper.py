import joblib
import numpy as np


class QuantileDemandModel:
    def __init__(self, models):
        """
        Args:
            models (dict): {quantile: trained LightGBM model}
        """
        self.models = models
        self.quantiles = sorted(models.keys())

    def predict(self, X):
        """
        Returns:
            dict: {quantile: predictions}
        """
        preds = {}

        for q, model in self.models.items():
            pred_log = model.predict(X)
            pred = np.expm1(pred_log)  # inverse log1p
            preds[q] = pred

        return preds

    def save(self, path):
        joblib.dump(self.models, path)

    @classmethod
    def load(cls, path):
        models = joblib.load(path)
        return cls(models)