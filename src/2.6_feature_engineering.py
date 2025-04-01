import pandas as pd
import joblib
import numpy as np

# Load raw data
df = pd.read_csv("../data/diabetic_data.csv")

# Drop high-missing or ID columns
df.drop(columns=[
    'encounter_id', 'patient_nbr', 'weight', 'payer_code',
    'medical_specialty', 'max_glu_serum', 'A1Cresult'
], inplace=True)

# Replace '?' with NaN and drop missing
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Encode age to ordinal midpoints
age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
df['age'] = df['age'].map(age_map)

# Readmission label
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Total prior visits
df['total_prev_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
df['is_frequent_user'] = (df['total_prev_visits'] > 5).astype(int)

# Binary med flags
df['meds_changed'] = (df['change'] == 'Ch').astype(int)
df['on_insulin'] = (df['insulin'] != 'No').astype(int)

# Medication count
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone']

df['num_meds_active'] = df[med_cols].apply(lambda row: (row != 'No').sum(), axis=1)

# Medications per day
df['meds_per_day'] = df['num_medications'] / (df['time_in_hospital'] + 1)

# Lab procedures per visit
df['labs_per_visit'] = df['num_lab_procedures'] / (df['total_prev_visits'] + 1)

# Diag group features
def map_diag(code):
    try:
        code = float(code)
        if (390 <= code <= 459) or (code == 785):
            return 'Circulatory'
        elif (460 <= code <= 519) or (code == 786):
            return 'Respiratory'
        elif (520 <= code <= 579) or (code == 787):
            return 'Digestive'
        elif 250 <= code < 251:
            return 'Diabetes'
        elif 800 <= code <= 999:
            return 'Injury'
        elif 710 <= code <= 739:
            return 'Musculoskeletal'
        elif 580 <= code <= 629 or code == 788:
            return 'Genitourinary'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except:
        return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col + '_group'] = df[col].apply(map_diag)

# Drop original diag cols
df.drop(columns=['diag_1', 'diag_2', 'diag_3'], inplace=True)

# One-hot encode new categorical features
cat_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Save full feature-engineered dataset
joblib.dump(df_encoded, '../data/df_engineered.pkl')

print(f"✅ Feature engineering complete. Final shape: {df_encoded.shape}")
