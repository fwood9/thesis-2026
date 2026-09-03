import os
import numpy as np
import pandas as pd
import joblib

from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, cohen_kappa_score,
                             average_precision_score)
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBClassifier

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, FixedLocator, NullLocator
import statsmodels.formula.api as smf

SEED = int(os.environ.get("SEED", 10))
BASE = "artifacts"  # fixed stage 1-2 inputs (never reseeded)
ART = os.path.join("artifacts", f"seed_{SEED}")  # per-seed outputs
os.makedirs(ART, exist_ok=True)

# Reading in the necessary files
split = joblib.load(os.path.join(BASE, "split.joblib"))
df_model = joblib.load(os.path.join(BASE, "df_model.joblib"))
X_train = split["X_train"]
X_test = split["X_test"]
y_train = split["y_train"]  # multiclass: 0=NO, 1=>30, 2=<30
y_test = split["y_test"]
numeric_cols = split["numeric_cols"]
categorical_cols = split["categorical_cols"]

# Drop sensitive features from the model inputs

SENSITIVE_PREFIXES = ("race_",)

sensitive_feats = [c for c in X_train.columns if c.startswith(SENSITIVE_PREFIXES)]
print("Dropping sensitive model features:", sensitive_feats)

X_train = X_train.drop(columns=sensitive_feats)
X_test = X_test.drop(columns=sensitive_feats)
numeric_cols = [c for c in numeric_cols if c not in sensitive_feats]
categorical_cols = [c for c in categorical_cols if c not in sensitive_feats]

# Rebuild the preprocessor
preprocess = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), numeric_cols),
    ('cat', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ]), categorical_cols),
])

# SMOTE-NC within race x class strata on the traning set
# Classes are balanced to 1:1:1 within each race group

# First need to impute NaN values
X_sm = X_train.copy()
if X_sm[numeric_cols].isna().any().any():
    X_sm[numeric_cols] = SimpleImputer(strategy="median").fit_transform(X_sm[numeric_cols])
if categorical_cols and X_sm[categorical_cols].isna().any().any():
    X_sm[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X_sm[categorical_cols])

cat_idx = [X_sm.columns.get_loc(c) for c in categorical_cols]

# Standardise the continuous columns before SMOTE-NC

binary_numeric = [c for c in numeric_cols if X_sm[c].dropna().nunique() <= 2]
cont_numeric = [c for c in numeric_cols if c not in binary_numeric]
smote_scaler = StandardScaler().fit(X_sm[cont_numeric])
X_sm_scaled = X_sm.copy()
X_sm_scaled[cont_numeric] = smote_scaler.transform(X_sm[cont_numeric])

# Race group labels for the training rows
train_race = df_model.loc[X_train.index, "race"].fillna("Missing").astype(str)

X_parts, y_parts = [], []
for g in sorted(train_race.unique()):
    mask = (train_race == g).values
    Xg, yg = X_sm_scaled.loc[mask], y_train.loc[mask]
    cc = yg.value_counts()

    if len(cc) < 2 or cc.min() < 2:
        # nothing sensible to resample - keep the group's rows as they are
        X_parts.append(Xg)
        y_parts.append(yg)
        print(f"race={g:<20} n={len(yg):>6}  kept as-is (too few minority rows)")
        continue

    k = int(min(5, cc.min() - 1))  # SMOTE needs k < smallest class count
    try:
        sampler = SMOTENC(categorical_features=cat_idx,
                          sampling_strategy="auto",
                          k_neighbors=k,
                          random_state=SEED)
        Xr, yr = sampler.fit_resample(Xg, yg)
    except ValueError:
        # tiny stratum where SMOTE still fails: duplicate rows instead
        sampler = RandomOverSampler(sampling_strategy="auto", random_state=SEED)
        Xr, yr = sampler.fit_resample(Xg, yg)

    X_parts.append(pd.DataFrame(Xr, columns=X_sm.columns))
    y_parts.append(pd.Series(yr, name=y_train.name))
    print(f"race={g:<20} n={len(yg):>6} -> {len(yr):>6}  (k_neighbors={k})")

X_train_sm = pd.concat(X_parts, ignore_index=True)
y_train_sm = pd.concat(y_parts, ignore_index=True).astype(int)

# Back to original units for the downstream pipeline
X_train_sm[cont_numeric] = smote_scaler.inverse_transform(
    X_train_sm[cont_numeric].astype(float))

print("Original train shape: ", X_train.shape)
print("Resampled train shape:", X_train_sm.shape)
print("Original class distribution: ")
print(y_train.value_counts(normalize=True).sort_index().round(3))
print("Resampled class distribution:")
print(y_train_sm.value_counts(normalize=True).sort_index().round(3))

# Cumulative binary targets from the resampled multiclass label
y_train_sm_ge1 = (y_train_sm >= 1).astype(int)  # readmitted at all
y_train_sm_ge2 = (y_train_sm >= 2).astype(int)  # readmitted <30

# Train the two cumulative XGBoost models

param_dist_xgb = {
    'clf__n_estimators': randint(200, 700),
    'clf__max_depth': randint(3, 8),
    'clf__learning_rate': loguniform(0.01, 0.3),
    'clf__subsample': uniform(0.6, 0.4),
    'clf__colsample_bytree': uniform(0.6, 0.4),
    'clf__min_child_weight': randint(1, 10),
    'clf__reg_lambda': loguniform(1e-2, 10),
}


def train_head(y_bin, scoring, label):
    pos = (y_bin == 1).sum()
    neg = (y_bin == 0).sum()
    spw = neg / pos
    print(f'scale_pos_weight ({label}):', round(spw, 2))

    pipe = Pipeline(steps=[
        ('preprocess', clone(preprocess)),
        ('clf', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=spw,
            n_jobs=-1,
            random_state=SEED,
            tree_method='hist',
        )),
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist_xgb,
        n_iter=40,
        scoring=scoring,
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        verbose=1,
        random_state=SEED,
    )
    search.fit(X_train_sm, y_bin)
    print(f'{label} best CV score:', round(search.best_score_, 4))
    print(f'{label} best params:', search.best_params_)
    return search.best_estimator_


best_ge1 = train_head(y_train_sm_ge1, 'roc_auc', 'Model 1 (y>=1)')
best_ge2 = train_head(y_train_sm_ge2, 'average_precision', 'Model 2 (y>=2)')

joblib.dump({"best_ge1": best_ge1, "best_ge2": best_ge2},
            os.path.join(ART, "models_fair.joblib"))

# Isotonic calibration on the training set

cal_ge1 = CalibratedClassifierCV(clone(best_ge1), method='isotonic', cv=5)
cal_ge2 = CalibratedClassifierCV(clone(best_ge2), method='isotonic', cv=5)
cal_ge1.fit(X_train_sm, y_train_sm_ge1)
cal_ge2.fit(X_train_sm, y_train_sm_ge2)

joblib.dump({"cal_ge1": cal_ge1, "cal_ge2": cal_ge2},
            os.path.join(ART, "calibrated_fair.joblib"))

# Predict on the untouched test set using the same metrics

p_ge1_test = cal_ge1.predict_proba(X_test)[:, 1]
p_ge2_test = cal_ge2.predict_proba(X_test)[:, 1]

p_class0 = 1 - p_ge1_test
p_class1 = p_ge1_test - p_ge2_test
p_class2 = p_ge2_test

y_proba_ordinal = np.vstack([p_class0, p_class1, p_class2]).T

violations = (p_ge2_test > p_ge1_test).sum()
print(f"\nMonotonicity violations before correction: {violations} / {len(p_ge1_test)} "
      f"({violations / len(p_ge1_test):.2%})")

y_proba_ordinal = np.clip(y_proba_ordinal, 0, None)
y_proba_ordinal = y_proba_ordinal / y_proba_ordinal.sum(axis=1, keepdims=True)
y_pred_ordinal = y_proba_ordinal.argmax(axis=1)

# Three-way ordinal prediction via cumulative-probability thresholds
y_true = y_test.values

# The target rates come from the training labels,
# so no held-out label informs the operating point.

rate_ge1 = (y_train >= 1).mean()
rate_ge2 = (y_train == 2).mean()

t1 = float(np.quantile(p_ge1_test, 1 - rate_ge1))
t2 = float(np.quantile(p_ge2_test, 1 - rate_ge2))
print(f"thresholds (test-ranked, train-rate)   t1 (>=1): {t1:.3f}   t2 (<30): {t2:.3f}")

y_pred_3way = np.where(p_ge2_test >= t2, 2,
                       np.where(p_ge1_test >= t1, 1, 0))

print("\npredicted distribution:", np.round(np.bincount(y_pred_3way, minlength=3) / len(y_true), 3))
print("actual    distribution:", np.round(np.bincount(y_true, minlength=3) / len(y_true), 3))

print("\n===== fair SMOTE model: threshold-decoded 3-way =====")
print("Classification report (0=NO, 1=>30, 2=<30):")
print(classification_report(y_true, y_pred_3way, digits=3,
                            target_names=["NO", ">30", "<30"]))
cm_3way = pd.DataFrame(confusion_matrix(y_true, y_pred_3way),
                       index=["NO", ">30", "<30"], columns=["NO", ">30", "<30"])
print("Confusion matrix (rows = true, cols = predicted):")
print(cm_3way)
cm_3way.to_csv(os.path.join(ART, "confusion_matrix_3way_fair.csv"))
print("Quadratic weighted kappa:",
      round(cohen_kappa_score(y_true, y_pred_3way, weights="quadratic"), 3))

print("\n===== fair SMOTE model: Frank-Hall argmax =====")
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
cm_ordinal.to_csv(os.path.join(ART, "confusion_matrix_ordinal_fair.csv"))

joblib.dump({
    "p_ge1_test": p_ge1_test, "p_ge2_test": p_ge2_test,
    "y_proba_ordinal": y_proba_ordinal, "y_pred_ordinal": y_pred_ordinal,
    "y_pred_3way": y_pred_3way, "t1": t1, "t2": t2,
}, os.path.join(ART, "predictions_fair.joblib"))
print("saved", os.path.join(ART, "predictions_fair.joblib"))

# Three-way fairness audits like before (race still used here as it is the audit only)

class_names = {0: "NO", 1: ">30", 2: "<30"}

age_bins = [0, 20, 40, 60, 90, 100]
age_labels = ["10-20", "20-40", "40-60", "60-90", "90-100"]
age_bucket = pd.cut(df_model.loc[X_test.index, "age_numeric"],
                    bins=age_bins, labels=age_labels, right=False, include_lowest=True)

audit = pd.DataFrame({
    "race": df_model.loc[X_test.index, "race"].fillna("Missing").astype(str).values,
    "gender": df_model.loc[X_test.index, "gender"].astype(str).values,
    "age": age_bucket.values,
    "y_true": y_test.values,
    "y_pred": y_pred_3way,
})
audit["race_age"] = audit["race"] + "_" + audit["age"].astype(str)


def class_metrics(sub, cls):
    yt = (sub["y_true"] == cls).astype(int).values
    yp = (sub["y_pred"] == cls).astype(int).values
    tp = int(((yp == 1) & (yt == 1)).sum());
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum());
    fn = int(((yp == 0) & (yt == 1)).sum())
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


for attr in ["race", "gender", "age", "race_age"]:
    print(f"\n===== three-way fairness by {attr} (fair SMOTE model) =====")
    result = fairness_by(attr).round(3)
    print(result)
    result.to_csv(os.path.join(ART, f"fairness_{attr}_fair.csv"), index=False)

# Logits plus the forest plot to examine bias by race

REFERENCE_RACE = "Caucasian"

d_logit = pd.DataFrame({
    "race": df_model.loc[X_test.index, "race"].fillna("Missing").astype(str).values,
    "flagged_ge1": (y_pred_3way >= 1).astype(int),
    "flagged_lt30": (y_pred_3way == 2).astype(int),
})


def tidy_logit(res):
    conf = res.conf_int()
    out = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "odds_ratio": np.exp(res.params.values),
        "or_lo": np.exp(conf[0].values),
        "or_hi": np.exp(conf[1].values),
        "p_value": res.pvalues.values,
    })
    out["term"] = out["term"].str.replace(
        r'C\(race, Treatment\(reference="' + REFERENCE_RACE + r'"\)\)\[T\.(.+)\]',
        r'race[\1]', regex=True)
    return out


def fit_and_report(outcome):
    formula = f'{outcome} ~ C(race, Treatment(reference="{REFERENCE_RACE}"))'
    res = smf.logit(formula, data=d_logit).fit(disp=0)
    table = tidy_logit(res)
    print(f"\n===== {outcome} ~ race (unadjusted, reference race: {REFERENCE_RACE}, fair SMOTE model) =====")
    print(table.round(4).to_string(index=False))
    print("\n--- race terms only ---")
    print(table[table["term"].str.startswith("race[")].round(4).to_string(index=False))
    table.to_csv(os.path.join(ART, f"logit_{outcome}_fair.csv"), index=False)
    return table


fit_and_report("flagged_ge1")
fit_and_report("flagged_lt30")

classes = {0: "NO", 1: ">30", 2: "<30"}
TEAL = {"NO": "#80CBC4", ">30": "#26A69A", "<30": "#00695C"}
offsets = {"NO": 0.24, ">30": 0.0, "<30": -0.24}

results = {}
for k, name in classes.items():
    d_logit["y"] = (y_pred_3way == k).astype(int)
    res = smf.logit(f'y ~ C(race, Treatment(reference="{REFERENCE_RACE}"))',
                    data=d_logit).fit(disp=0)
    conf = res.conf_int()
    t = pd.DataFrame({
        "term": res.params.index,
        "odds_ratio": np.exp(res.params.values),
        "or_lo": np.exp(conf[0].values),
        "or_hi": np.exp(conf[1].values),
    })
    t = t[t["term"].str.contains(r"\[T\.")].copy()
    t["label"] = t["term"].str.replace(r'.*\[T\.(.+)\]', r'\1', regex=True)
    results[name] = t.set_index("label")

# fixed race order (largest group first) so every variant's plot matches
race_counts = d_logit["race"].value_counts()
races = [r for r in race_counts.index if r != REFERENCE_RACE and r in results["<30"].index]

fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(0.8, 1.25, color="grey", alpha=0.08, zorder=0)  # negligible band
for yi in range(len(races)):
    ax.axhline(yi, color="0.9", lw=0.8, zorder=0)  # row guides

for name in classes.values():
    t = results[name].reindex(races)
    for yi, (_, row) in enumerate(t.iterrows()):
        y = yi + offsets[name]
        ax.plot([row["or_lo"], row["or_hi"]], [y, y], color=TEAL[name],
                lw=2.4, solid_capstyle="round", zorder=2)
        ax.plot(row["odds_ratio"], y, "o", color=TEAL[name], ms=6.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)

ax.axvline(1.0, color="black", ls="--", lw=1, zorder=1)
ax.set_xscale("log")
ax.set_xlim(0.2, 5.0)  # fixed, identical on every plot
ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 2.0, 5.0]))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks(range(len(races)))
ax.set_yticklabels(races)
ax.set_ylim(-0.6, len(races) - 0.4)
ax.set_xlabel("Odds ratio vs Caucasian (log scale)")
ax.set_title("Race allocation bias by predicted outcome (fair SMOTE model)")
ax.grid(axis="x", ls=":", alpha=0.4)

handles = [Line2D([0], [0], color=TEAL[n], lw=2.4, marker="o", label=n)
           for n in classes.values()]
ax.legend(handles=handles, title="Predicted outcome", loc="upper left",
          bbox_to_anchor=(1.02, 1), frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(ART, "logit_forest_3way_fair.png"), dpi=200, bbox_inches="tight")
print("\nsaved", os.path.join(ART, "logit_forest_3way_fair.png"))