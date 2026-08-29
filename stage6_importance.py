import os
import numpy as np
import pandas as pd
import joblib

from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, average_precision_score

SEED = int(os.environ.get("SEED", 10))
BASE = "artifacts"                               # fixed stage 1-2 inputs (never reseeded)
ART = os.path.join("artifacts", f"seed_{SEED}")  # per-seed outputs
os.makedirs(ART, exist_ok=True)

split = joblib.load(os.path.join(BASE, "split.joblib"))
calibrated = joblib.load(os.path.join(ART, "calibrated.joblib"))
predictions = joblib.load(os.path.join(ART, "predictions.joblib"))

X_test = split["X_test"]
y_test = split["y_test"]
cal_ge1 = calibrated["cal_ge1"]
cal_ge2 = calibrated["cal_ge2"]
t1 = predictions["t1"]
t2 = predictions["t2"]


class FrankHallOrdinalClassifier:
    """Combines the two calibrated cumulative models into one ordinal classifier."""

    def __init__(self, model_ge1, model_ge2, t1=0.5, t2=0.5):
        self.model_ge1 = model_ge1
        self.model_ge2 = model_ge2
        self.t1 = t1  # cutoff on P(y>=1)  ">= >30"
        self.t2 = t2  # cutoff on P(y>=2)  "<30"

    def fit(self, X, y=None):
        return self  # models already fitted; so here only for the sklearn interface

    def _cumulative(self, X):
        p_ge1 = self.model_ge1.predict_proba(X)[:, 1]
        p_ge2 = self.model_ge2.predict_proba(X)[:, 1]
        return p_ge1, p_ge2

    def predict_proba(self, X):
        p_ge1, p_ge2 = self._cumulative(X)
        proba = np.vstack([1 - p_ge1, p_ge1 - p_ge2, p_ge2]).T
        proba = np.clip(proba, 0, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        p_ge1, p_ge2 = self._cumulative(X)
        # most-severe threshold that fires wins (same rule as the 3-way decoding cell)
        return np.where(p_ge2 >= self.t2, 2,
                        np.where(p_ge1 >= self.t1, 1, 0))

    def get_params(self, deep=True):
        return {"model_ge1": self.model_ge1, "model_ge2": self.model_ge2,
                "t1": self.t1, "t2": self.t2}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def score(self, X, y):
        # ordinal agreement at the fixed threshold operating point
        return cohen_kappa_score(y, self.predict(X), weights="quadratic")


# Reuse the SAME base-rate thresholds computed in the 3-way decoding cell
combined_model = FrankHallOrdinalClassifier(cal_ge1, cal_ge2, t1=t1, t2=t2)

perm_result = permutation_importance(
    combined_model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=SEED,
    n_jobs=-1,
)

overall_importance = (
    pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    })
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)

overall_importance.to_csv(os.path.join(ART, "overall_importance.csv"), index=False)

# Permutation importance of the race identifiers for the 3-way model
race_features = [c for c in X_test.columns if c.startswith("race_")]
race_idx = [list(X_test.columns).index(c) for c in race_features]

race_importance = (
    pd.DataFrame({
        "feature": race_features,
        "importance_mean": perm_result.importances_mean[race_idx],
        "importance_std": perm_result.importances_std[race_idx],
    })
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)

race_importance.to_csv(os.path.join(ART, "race_importance.csv"), index=False)

# List of features that drive the <30 classification using permutation
def lt30_ap_scorer(estimator, X, y):
    p2 = estimator.predict_proba(X)[:, 1]
    return average_precision_score((np.asarray(y) == 2).astype(int), p2)


perm_lt30 = permutation_importance(
    cal_ge2,
    X_test,
    y_test,
    scoring=lt30_ap_scorer,
    n_repeats=10,
    random_state=SEED,
    n_jobs=-1
)

lt30_importance = (
    pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': perm_lt30.importances_mean,
        'importance_std': perm_lt30.importances_std,
    })
    .sort_values('importance_mean', ascending=False)
    .reset_index(drop=True)
)

lt30_importance.to_csv(os.path.join(ART, "lt30_importance.csv"), index=False)
