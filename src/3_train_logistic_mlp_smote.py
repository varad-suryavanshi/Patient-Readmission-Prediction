import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load reduced data
X_train = joblib.load('../data/X_train_reduced.pkl')
X_test = joblib.load('../data/X_test_reduced.pkl')
y_train = joblib.load('../data/y_train.pkl')
y_test = joblib.load('../data/y_test.pkl')

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Standardize for linear/deep models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

print("✅ Data scaled and SMOTE applied.")

# ======================
# Logistic Regression
# ======================
print("\n🔹 Training: Logistic Regression")
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logreg.fit(X_train_scaled, y_train_res)
log_preds = logreg.predict(X_test_scaled)
log_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Logistic Regression Performance:")
print(classification_report(y_test, log_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_preds))
print("ROC AUC Score:", roc_auc_score(y_test, log_proba))

# ======================
# MLP Classifier
# ======================
print("\n🔹 Training: MLP (Neural Network)")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                    max_iter=100, random_state=42)
mlp.fit(X_train_scaled, y_train_res)
mlp_preds = mlp.predict(X_test_scaled)
mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]

print("\n📊 MLP Performance:")
print(classification_report(y_test, mlp_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, mlp_preds))
print("ROC AUC Score:", roc_auc_score(y_test, mlp_proba))

# Save best model
joblib.dump(logreg, '../data/readmission_model_logreg.pkl')
joblib.dump(mlp, '../data/readmission_model_mlp.pkl')
joblib.dump(scaler, '../data/scaler.pkl')

print("\n✅ Models and scaler saved.")
