import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

# Loading data
df_model = joblib.load(os.path.join(ART, "df_model.joblib"))

# Labelling the target column
target = 'readmit_class'

# Organising the feature columns
exclude_from_features = [
    'readmitted',
    'readmit_class',
    'race',
    'gender',
    'age',
    'max_glu_serum',
    'A1Cresult',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'repaglinide',
    'nateglinide',
    'chlorpropamide',
    'acetohexamide',
    'glyburide',
    'tolbutamide',
    'rosiglitazone',
    'acarbose',
    'miglitol',
    'troglitazone',
    'tolazamide',
    'glyburide-metformin',
    'glipizide-metformin',
    'glimepiride-pioglitazone',
    'metformin-rosiglitazone',
    'metformin-pioglitazone',
]

# Excluding the target variable and other non-feature columns from the feature set
features = [c for c in df_model.columns if c not in exclude_from_features]

# Splitting the data
X = df_model[features].copy()
y = df_model[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,   # 70/30 split
    random_state=10,
    stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Class distribution (train):")
print(y_train.value_counts(normalize=True).sort_index().round(3))
print("Class distribution (test):")
print(y_test.value_counts(normalize=True).sort_index().round(3))

# Differentiating between numerical and categorical variables
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Column transformers for the preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ]
)

# Build the two cumulative binary targets

# 1 = readmitted at all ('>30' or ',30'), 0 = 'NO'
y_train_ge1 = (y_train >= 1).astype(int)
y_test_ge1 = (y_test >= 1).astype(int)

# 1 = readmitted '<30', 0 = otherwise
y_train_ge2 = (y_train >= 2).astype(int)
y_test_ge2 = (y_test >= 2).astype(int)

# Save outputs
joblib.dump({
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
    "y_train_ge1": y_train_ge1, "y_test_ge1": y_test_ge1,
    "y_train_ge2": y_train_ge2, "y_test_ge2": y_test_ge2,
    "preprocess": preprocess,
    "features": features,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
}, os.path.join(ART, "split.joblib"))
