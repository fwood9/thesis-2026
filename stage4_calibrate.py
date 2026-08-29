import os
import joblib

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

SEED = int(os.environ.get("SEED", 10))
BASE = "artifacts"                               # fixed stage 1-2 inputs (never reseeded)
ART = os.path.join("artifacts", f"seed_{SEED}")  # per-seed outputs
os.makedirs(ART, exist_ok=True)

# Reading in the necessary files
split = joblib.load(os.path.join(BASE, "split.joblib"))
models = joblib.load(os.path.join(ART, "models.joblib"))

X_train = split["X_train"]
y_train_ge1 = split["y_train_ge1"]
y_train_ge2 = split["y_train_ge2"]
best_ge1 = models["best_ge1"]
best_ge2 = models["best_ge2"]

# Calibrate the cumulative models on just the training set
# Wraps the tuned pipelines such that all downstream probabilities are calculated
cal_ge1 = CalibratedClassifierCV(clone(best_ge1), method='isotonic', cv=5)
cal_ge2 = CalibratedClassifierCV(clone(best_ge2), method='isotonic', cv=5)
cal_ge1.fit(X_train, y_train_ge1)
cal_ge2.fit(X_train, y_train_ge2)

# Saving the calibrated files
joblib.dump({"cal_ge1": cal_ge1, "cal_ge2": cal_ge2},
            os.path.join(ART, "calibrated.joblib"))
print("saved", os.path.join(ART, "calibrated.joblib"))