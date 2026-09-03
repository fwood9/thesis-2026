import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, cohen_kappa_score

SEED = int(os.environ.get("SEED", 10))
BASE = "artifacts"  # fixed stage 1-2 inputs (never reseeded)
ART = os.path.join("artifacts", f"seed_{SEED}")  # per-seed outputs
os.makedirs(ART, exist_ok=True)

# Reading in the necessary files
split = joblib.load(os.path.join(BASE, "split.joblib"))
calibrated = joblib.load(os.path.join(ART, "calibrated.joblib"))

X_test = split["X_test"]
y_test = split["y_test"]
X_train = split["X_train"]
y_train = split["y_train"]
cal_ge1 = calibrated["cal_ge1"]
cal_ge2 = calibrated["cal_ge2"]

# Combine the two binary models into a 3-class probability
p_ge1_test = cal_ge1.predict_proba(X_test)[:, 1]
p_ge2_test = cal_ge2.predict_proba(X_test)[:, 1]

p_class0 = 1 - p_ge1_test
p_class1 = p_ge1_test - p_ge2_test
p_class2 = p_ge2_test

y_proba_ordinal = np.vstack([p_class0, p_class1, p_class2]).T

# How often is monotonicity (P(y>=2) <= P(y>=1)) violated before correction?
violations = (p_ge2_test > p_ge1_test).sum()
print(f"Monotonicity violations before correction: {violations} / {len(p_ge1_test)} "
      f"({violations / len(p_ge1_test):.2%})")

# Clip negative probabilities to 0, then renormalise each row to sum to 1
y_proba_ordinal = np.clip(y_proba_ordinal, 0, None)
row_sums = y_proba_ordinal.sum(axis=1, keepdims=True)
y_proba_ordinal = y_proba_ordinal / row_sums

y_pred_ordinal = y_proba_ordinal.argmax(axis=1)

# Three-way ordinal prediction via cumulative-probability thresholds

y_true = y_test.values
p1 = p_ge1_test
p2 = p_ge2_test

# The target rates come from the training labels,
# so no held-out label informs the operating point.

rate_ge1 = (y_train >= 1).mean()  # (any readmission, training prevalence)
rate_ge2 = (y_train == 2).mean()  # (<30, training prevalence)

t1 = float(np.quantile(p1, 1 - rate_ge1))  # cutoff for ">= >30"
t2 = float(np.quantile(p2, 1 - rate_ge2))  # cutoff for "<30"
print(f"thresholds (test-ranked, train-rate)   t1 (>=1): {t1:.3f}   t2 (<30): {t2:.3f}")

# Ordinal decision: most-severe threshold that fires wins
y_pred_3way = np.where(p2 >= t2, 2,
                       np.where(p1 >= t1, 1, 0))

print("\npredicted distribution:", np.round(np.bincount(y_pred_3way, minlength=3) / len(y_true), 3))
print("actual    distribution:", np.round(np.bincount(y_true, minlength=3) / len(y_true), 3))

print("\nClassification report (0=NO, 1=>30, 2=<30):")
print(classification_report(y_true, y_pred_3way, digits=3,
                            target_names=["NO", ">30", "<30"]))
cm_3way = pd.DataFrame(confusion_matrix(y_true, y_pred_3way),
                       index=["NO", ">30", "<30"], columns=["NO", ">30", "<30"])
print("Confusion matrix (rows = true, cols = predicted):")
print(cm_3way)
cm_3way.to_csv(os.path.join(ART, "confusion_matrix_3way.csv"))
print("Quadratic weighted kappa:",
      round(cohen_kappa_score(y_true, y_pred_3way, weights="quadratic"), 3))

# Evaluate the Frank-Hall model on the test set
roc_auc_ordinal = roc_auc_score(y_test, y_proba_ordinal, multi_class="ovr", average="weighted")
qwk = cohen_kappa_score(y_test, y_pred_ordinal, weights="quadratic")

print(f"ROC-AUC (weighted OVR): {roc_auc_ordinal:.3f}")
print(f"Quadratic weighted kappa: {qwk:.3f}")

print("\nClassification report (0=NO, 1=>30, 2=<30):")
print(classification_report(y_test, y_pred_ordinal, digits=3,
                            target_names=["NO", ">30", "<30"]))

cm_ordinal = pd.DataFrame(confusion_matrix(y_test, y_pred_ordinal),
                          index=["NO", ">30", "<30"], columns=["NO", ">30", "<30"])
print("\nConfusion matrix (rows = true, cols = predicted):")
print(cm_ordinal)
cm_ordinal.to_csv(os.path.join(ART, "confusion_matrix_ordinal.csv"))

joblib.dump({
    "p_ge1_test": p_ge1_test, "p_ge2_test": p_ge2_test,
    "y_proba_ordinal": y_proba_ordinal, "y_pred_ordinal": y_pred_ordinal,
    "y_pred_3way": y_pred_3way, "t1": t1, "t2": t2,
}, os.path.join(ART, "predictions.joblib"))
print("saved", os.path.join(ART, "predictions.joblib"))