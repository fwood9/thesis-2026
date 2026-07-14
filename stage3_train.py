import os
import joblib

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBClassifier

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

# Reading in the necessary files
split = joblib.load(os.path.join(ART, "split.joblib"))
X_train = split["X_train"]
y_train_ge1 = split["y_train_ge1"]
y_train_ge2 = split["y_train_ge2"]
preprocess = split["preprocess"]

# Train model 1: P(y >= 1)
pos_ge1 = (y_train_ge1 == 1).sum()
neg_ge1 = (y_train_ge1 == 0).sum()
scale_pos_weight_ge1 = neg_ge1 / pos_ge1
print('scale_pos_weight (Model 1):', round(scale_pos_weight_ge1, 2))

pipe_ge1 = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_ge1,
        n_jobs=-1,
        random_state=10,
        tree_method='hist',
    )),
])

param_dist_xgb = {
    'clf__n_estimators': randint(200, 700),
    'clf__max_depth': randint(3, 8),
    'clf__learning_rate': loguniform(0.01, 0.3),
    'clf__subsample': uniform(0.6, 0.4),
    'clf__colsample_bytree': uniform(0.6, 0.4),
    'clf__min_child_weight': randint(1, 10),
    'clf__reg_lambda': loguniform(1e-2, 10),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

search_ge1 = RandomizedSearchCV(
    estimator=pipe_ge1,
    param_distributions=param_dist_xgb,
    n_iter=40,
    scoring='roc_auc',
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=10,
)

search_ge1.fit(X_train, y_train_ge1)
best_ge1 = search_ge1.best_estimator_
print('Model 1 best CV score:', round(search_ge1.best_score_, 4))
print('Model 1 best CV score:', search_ge1.best_params_)

# Train model 2: P(y >= 2)
pos_ge2 = (y_train_ge2 == 1).sum()
neg_ge2 = (y_train_ge2 == 0).sum()
scale_pos_weight_ge2 = neg_ge2 / pos_ge2
print('scale_pos_weight (Model 2):', round(scale_pos_weight_ge2, 2))

pipe_ge2 = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_ge2,
        n_jobs=-1,
        random_state=10,
        tree_method='hist',
    )),
])

param_dist_xgb = {
    'clf__n_estimators': randint(200, 700),
    'clf__max_depth': randint(3, 8),
    'clf__learning_rate': loguniform(0.01, 0.3),
    'clf__subsample': uniform(0.6, 0.4),
    'clf__colsample_bytree': uniform(0.6, 0.4),
    'clf__min_child_weight': randint(1, 10),
    'clf__reg_lambda': loguniform(1e-2, 10),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

search_ge2 = RandomizedSearchCV(
    estimator=pipe_ge2,
    param_distributions=param_dist_xgb,
    n_iter=40,
    scoring='average_precision',
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=10,
)

search_ge2.fit(X_train, y_train_ge2)
best_ge2 = search_ge2.best_estimator_
print('Model 2 best CV score:', round(search_ge2.best_score_, 4))
print('Model 2 best CV score:', search_ge2.best_params_)

# Saving the outputs
joblib.dump({"best_ge1": best_ge1, "best_ge2": best_ge2},
            os.path.join(ART, "models.joblib"))
print("saved", os.path.join(ART, "models.joblib"))
