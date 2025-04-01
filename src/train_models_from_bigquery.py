import pandas as pd
import joblib
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 📡 Load from BigQuery
client = bigquery.Client(project="healthcare-readmission-455023")
query = """
SELECT *
FROM `healthcare-readmission-455023.readmission_dataset.feature_engineered_readmission`
"""
df = client.query(query).to_dataframe()
print(f"✅ Loaded from BigQuery: {df.shape}")

# 🎯 Split features and target
X = df.drop(columns=['readmitted', 'readmit_flag'])
y = df['readmit_flag']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Clean column names for XGBoost compatibility
X.columns = X.columns.str.replace(r"[<>[\]]", "", regex=True)

# Ensure numeric dtype
X = X.astype('float32')




# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ⚖️ Balance classes with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 🔁 Standardize for LogReg
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 🧠 Logistic Regression Model
# ----------------------------
print("\n🔹 Logistic Regression")
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logreg.fit(X_train_scaled, y_train_res)
log_preds = logreg.predict(X_test_scaled)
log_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("📊 Logistic Regression:")
print(classification_report(y_test, log_preds))
print("ROC AUC:", roc_auc_score(y_test, log_proba))

# ----------------------------
# 🌲 XGBoost Model
# ----------------------------
print("\n🔹 XGBoost")
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)
xgb.fit(X_train_res, y_train_res)
xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

print("📊 XGBoost:")
print(classification_report(y_test, xgb_preds))
print("ROC AUC:", roc_auc_score(y_test, xgb_proba))

# 💾 Save models
joblib.dump(logreg, 'logreg_bq.pkl')
joblib.dump(xgb, 'xgb_bq.pkl')
joblib.dump(scaler, 'scaler_bq.pkl')
joblib.dump(X.columns.tolist(), "model_features.pkl")
print("\n✅ Models saved: logreg_bq.pkl, xgb_bq.pkl, scaler_bq.pkl")
