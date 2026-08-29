import os
import numpy as np
import pandas as pd
import joblib

SEED = int(os.environ.get("SEED", 10))
BASE = "artifacts"                               # fixed stage 1-2 inputs (never reseeded)
ART = os.path.join("artifacts", f"seed_{SEED}")  # per-seed outputs
os.makedirs(ART, exist_ok=True)

df_model = joblib.load(os.path.join(BASE, "df_model.joblib"))
split = joblib.load(os.path.join(BASE, "split.joblib"))
predictions = joblib.load(os.path.join(ART, "predictions.joblib"))

X_test = split["X_test"]
y_test = split["y_test"]
y_pred_3way = predictions["y_pred_3way"]

# Three-way fairness audit
class_names = {0: "NO", 1: ">30", 2: "<30"}

# Custom age buckets: 10-20, 20-40, 40-60, 60-90, 90-100
age_bins = [0, 20, 40, 60, 90, 100]
age_labels = ["10-20", "20-40", "40-60", "60-90", "90-100"]
age_bucket = pd.cut(df_model.loc[X_test.index, "age_numeric"],
                    bins=age_bins, labels=age_labels, right=False, include_lowest=True)

audit = pd.DataFrame({
    "race": df_model.loc[X_test.index, "race"].fillna("Missing").astype(str).values,
    "gender": df_model.loc[X_test.index, "gender"].astype(str).values,
    "age": age_bucket.values,  # Order categorical - keeps bucket order
    "y_true": y_test.values,
    "y_pred": y_pred_3way,
})

audit["race_age"] = audit["race"] + "_" + audit["age"].astype(str)


def class_metrics(sub, cls):
    yt = (sub["y_true"] == cls).astype(int).values
    yp = (sub["y_pred"] == cls).astype(int).values
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    P, N, n = tp + fn, tn + fp, len(sub)
    d = lambda a, b: a / b if b else np.nan
    base, sel = d(P, n), d(tp + fp, n)
    return dict(n=n, base_rate=base, sel_rate=sel, flag_ratio=d(sel, base),
                TPR=d(tp, P), FPR=d(fp, N), PPV=d(tp, tp + fp))


def fairness_by(attr, min_n=50):
    rows = []
    for g, sub in audit.groupby(attr, observed=True):
        if len(sub) < min_n:
            continue
        for cls in [0, 1, 2]:
            m = class_metrics(sub, cls)
            m[attr] = g
            m["class"] = class_names[cls]
            rows.append(m)
    cols = [attr, "class", "n", "base_rate", "sel_rate", "flag_ratio", "TPR", "FPR", "PPV"]
    return pd.DataFrame(rows)[cols]


print("\n===== three-way fairness by race_age =====")
result = fairness_by("race_age").round(3)
print(result)
result.to_csv(os.path.join(ART, "fairness_race_age.csv"), index=False)


