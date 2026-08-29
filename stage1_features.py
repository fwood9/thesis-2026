import os
import re
import pandas as pd
import numpy as np
import joblib

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

# Reading in the dataset
df = pd.read_csv('diabetic_data.csv')
print(df.shape)

# Removing encounters that end in death/ hospice as they cannot be readmitted
df = df.replace('?', np.nan)
df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])].copy()

# Dropping any duplicate values by keeping the first
if {'patient_nbr', 'encounter_id'}.issubset(df.columns):
    df = df.sort_values(['patient_nbr', 'encounter_id'])
    df = df.drop_duplicates(subset='patient_nbr', keep='first')

# Dropping the unwanted columns
df.drop(columns=[
    'weight',   # Missing Values
    'payer_code',  # Missing Values
    'encounter_id',  # Visit ID
    'patient_nbr',  # Patient ID
    'examide',  # All 'No'
    'citoglipton',  # All 'No'
], inplace=True)

# New df shape
print(df.shape)

# Creating binary features labelling what diagnosis a patient might have
def map_icd9_category(code):
    if pd.isna(code):
        return np.nan
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        code_num = float(code)
    except ValueError:
        return "Other"
    if 250 <= code_num < 251:            return "Diabetes"
    if (390 <= code_num <= 459) or code_num == 785:  return "Circulatory"
    if (460 <= code_num <= 519) or code_num == 786:  return "Respiratory"
    if (520 <= code_num <= 579) or code_num == 787:  return "Digestive"
    if 800 <= code_num <= 999:           return "Injury"
    if 710 <= code_num <= 739:           return "Musculoskeletal"
    if (580 <= code_num <= 629) or code_num == 788:  return "Genitourinary"
    if 140 <= code_num <= 239:           return "Cancer"
    return "Other"


diag_raw_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
cats = df[diag_raw_cols].applymap(map_icd9_category)

diag_categories = ["Circulatory", "Respiratory", "Digestive", "Diabetes",
                   "Injury", "Musculoskeletal", "Genitourinary", "Cancer", "Other"]

df["n_distinct_diag_categories"] = cats.nunique(axis=1)
df["diabetes_primary_diagnosis"] = (cats["diag_1"] == "Diabetes").astype(int)
for category in diag_categories:
    df[f"has_{category.lower()}"] = (cats == category).any(axis=1).astype(int)

df.drop(columns=diag_raw_cols, inplace=True)

# Creating 11 new binary indicators for medical specialty
# Keeping the top 10 and a new indicator which includes the entries with missing values
if "medical_specialty" in df.columns:
    spec = df["medical_specialty"].replace("?", np.nan)
    top10 = spec.value_counts().head(10).index

    slug = lambda x: re.sub(r"[^0-9a-zA-Z]+", "_", str(x)).strip("_")
    for s in top10:
        df[f"spec_{slug(s)}"] = (spec == s).astype(int)
    df["spec_other"] = (~spec.isin(top10)).astype(int)

    df.drop(columns="medical_specialty", inplace=True)

# Mapping the age midpoint number of each age bracket
age_map = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

df["age_numeric"] = df["age"].map(age_map)

# Creating medication counts
med_status_cols = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    "insulin",
]

med_cols_present = [c for c in med_status_cols if c in df.columns]

df["diabetes_med_count"] = (df[med_cols_present] != "No").sum(axis=1)
df["total_med_changes"] = (df[med_cols_present].isin(['Up', 'Down'])).sum(axis=1)

# Aggregate the visit and procedure counts
df["total_visits"] = (
    df["number_outpatient"].fillna(0)
    + df["number_emergency"].fillna(0)
    + df["number_inpatient"].fillna(0)
)

df['sum_emergency_and_inpatient'] = (
    df['number_emergency'].fillna(0)
    + df['number_inpatient'].fillna(0)
)

df["total_procedures"] = (
    df["num_lab_procedures"].fillna(0)
    + df["num_procedures"].fillna(0)
)

df['medical_burden'] = (
    df['total_procedures'].fillna(0)
    + df['number_diagnoses'].fillna(0)
)

# Severity indicators
df['many_medications'] = (df['num_medications'] >= 5).astype(int)
df['long_stay'] = (df['time_in_hospital'] >= 7).astype(int)

# Features from A1Cresult and max_glu_serum
df["a1c_measured"] = df["A1Cresult"].notna().astype(int)
df["glu_measured"] = df["max_glu_serum"].notna().astype(int)


def map_a1c_level(x):
    if pd.isna(x) or x == "None":
        return 0
    if x == "Norm":
        return 1
    if x == ">7":
        return 2
    if x == ">8":
        return 3
    return 0


df["a1c_level"] = df["A1Cresult"].apply(map_a1c_level)


def map_glu_level(x):
    if pd.isna(x) or x == "None":
        return 0
    if x == "Norm":
        return 1
    if x == ">200":
        return 2
    if x == ">300":
        return 3
    return 0


df["glu_level"] = df["max_glu_serum"].apply(map_glu_level)

df["a1c_high"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
df["glu_high"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)

# Admission / discharge context flags (ID meanings from IDS_mapping.csv)
df["is_emergency_admission"] = df["admission_type_id"].isin({1, 7}).astype(int)
df["is_unknown_discharge"] = df["discharge_disposition_id"].isin({9, 18, 25, 26}).astype(int)
df["discharged_to_another_facility"] = (~df["discharge_disposition_id"].isin({1, 6, 8, 9, 18, 25, 26})).astype(int)
df["admitted_from_er"] = df["admission_source_id"].isin({7}).astype(int)

# Binary features for Race and Gender
race_clean = df["race"].replace("?", np.nan)
for r in race_clean.dropna().unique():
    df[f"race_{r}"] = (race_clean == r).astype(int)
df["race_Missing"] = race_clean.isna().astype(int)

df["is_male"] = (df["gender"] == "Male").astype(int)
df["is_female"] = (df["gender"] == "Female").astype(int)

# Duplicating df to help with any future analysis needed
df_model = df.copy()

# Creating the multiclass format
# 0 = 'No'
# 1 = '>30'
# 2 = '<30'

readmit_mapping = {'NO': 0, '>30': 1, '<30': 2}
df_model['readmitted'] = df_model['readmitted'].astype(str).str.strip()
df_model['readmit_class'] = df_model['readmitted'].map(readmit_mapping)

# Saving outputs
joblib.dump(df, os.path.join(ART, "df_premodel.joblib"))
joblib.dump(df_model, os.path.join(ART, "df_model.joblib"))