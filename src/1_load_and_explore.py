import pandas as pd

# Load datasets
data_path = "../data/diabetic_data.csv"
ids_map_path = "../data/IDS_mapping.csv"

df = pd.read_csv(data_path)
ids_map = pd.read_csv(ids_map_path)

# Show basic info
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing value count:\n", df.isnull().sum())

# View unique values of key columns
print("\nReadmission value counts:\n", df['readmitted'].value_counts())
print("\nEncounter types:\n", df['admission_type_id'].value_counts())
print("\nDiagnosis columns:\n", df[['diag_1', 'diag_2', 'diag_3']].head())

# Preview mapping table
print("\nICD Code Mapping Sample:\n", ids_map.head())