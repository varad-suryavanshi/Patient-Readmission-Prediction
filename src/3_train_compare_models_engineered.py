import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load engineered dataset
df = joblib.load('../data/df_engineered.pkl')

# Split features/target
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# SMOTE for balanced training
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Standardize for LogReg + MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Dataset ready. Train shape: {X_train_res.shape}")

# -------------------
# Logistic Regression
# -------------------
print("\n🔹 Logistic Regression")
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logreg.fit(X_train_scaled, y_train_res)
log_preds = logreg.predict(X_test_scaled)
log_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, log_preds))
print("ROC AUC Score:", roc_auc_score(y_test, log_proba))

# -------------------
# MLP Classifier
# -------------------
print("\n🔹 MLP Classifier")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=150,
                    random_state=42, early_stopping=True)
mlp.fit(X_train_scaled, y_train_res)
mlp_preds = mlp.predict(X_test_scaled)
mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, mlp_preds))
print("ROC AUC Score:", roc_auc_score(y_test, mlp_proba))

# -------------------
# XGBoost
# -------------------
print("\n🔹 XGBoost Classifier")
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

print("Classification Report:")
print(classification_report(y_test, xgb_preds))
print("ROC AUC Score:", roc_auc_score(y_test, xgb_proba))

# Save all models + scaler
joblib.dump(logreg, '../data/logreg_engineered.pkl')
joblib.dump(mlp, '../data/mlp_engineered.pkl')
joblib.dump(xgb, '../data/xgb_engineered.pkl')
joblib.dump(scaler, '../data/scaler_engineered.pkl')

print("\n✅ All models trained and saved.")
