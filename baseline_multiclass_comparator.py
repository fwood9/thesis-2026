import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

# Config
BASE = "artifacts"
SPLIT_PATH = os.path.join(BASE, "split.joblib")
SEEDS = [4, 8, 9, 10, 30, 64, 67, 88, 117, 149]
N_ITER = 40
CV_FOLDS = 5

# Load the fixed split

split = joblib.load(SPLIT_PATH)

X_train = split["X_train"]
X_test = split["X_test"]
y_train = split["y_train"]
y_test = split["y_test"]

numeric_cols = list(split["numeric_cols"])
categorical_cols = list(split["categorical_cols"])

# Per-sample weights, computed once from training labels
sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Hyperparameter distributions

param_distributions = {
    "clf__n_estimators": randint(200, 800),
    "clf__max_depth": randint(3, 9),
    "clf__learning_rate": uniform(0.01, 0.29),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__min_child_weight": randint(1, 10),
    "clf__reg_lambda": uniform(0.0, 5.0),
}


def build_preprocess():
    return ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),

        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])


def run_seed(seed):
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("preprocess", build_preprocess()),
        ("clf", clf),
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="roc_auc_ovr_weighted",
        cv=StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=seed
        ),
        random_state=seed,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    search.fit(
        X_train,
        y_train,
        clf__sample_weight=sample_weight
    )

    proba = search.best_estimator_.predict_proba(X_test)

    auc = roc_auc_score(
        y_test,
        proba,
        multi_class="ovr",
        average="weighted"
    )

    return auc



# Run all seeds and report AUC

def main():
    rows = []

    for seed in SEEDS:
        auc = run_seed(seed)
        rows.append({
            "seed": seed,
            "AUC": auc
        })

        print(f"seed {seed}: AUC={auc:.3f}")

    res = pd.DataFrame(rows)

    res.to_csv("baseline_multiclass_per_seed.csv", index=False)

    print("\nSaved baseline_multiclass_per_seed.csv")


if __name__ == "__main__":
    main()

