import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load reduced data
X_train = joblib.load('../data/X_train_reduced.pkl')
X_test = joblib.load('../data/X_test_reduced.pkl')
y_train = joblib.load('../data/y_train.pkl')
y_test = joblib.load('../data/y_test.pkl')

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Clean feature names (again)
X_train_res.columns = X_train_res.columns.str.replace('[<>\[\]]', '', regex=True)
X_test.columns = X_test.columns.str.replace('[<>\[\]]', '', regex=True)

# Adjust for imbalance
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

print("\n🔹 Training: XGBoost with Top 100 Features + SMOTE")
xgb_model.fit(X_train_res, y_train_res)

# Evaluate
preds = xgb_model.predict(X_test)
proba = xgb_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

print("ROC AUC Score:", roc_auc_score(y_test, proba))

# Save model
joblib.dump(xgb_model, '../data/readmission_model_xgboost_reduced.pkl')
print("\n✅ Model saved as 'readmission_model_xgboost_reduced.pkl'")
