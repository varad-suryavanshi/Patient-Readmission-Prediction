import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load preprocessed data
X_train = joblib.load('../data/X_train.pkl')
X_test = joblib.load('../data/X_test.pkl')
y_train = joblib.load('../data/y_train.pkl')
y_test = joblib.load('../data/y_test.pkl')

# Apply SMOTE on training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("✅ Applied SMOTE:")
print("Resampled class distribution:\n", y_train_res.value_counts())

def train_and_evaluate(model, name):
    print(f"\n🔹 Training: {name}")
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("ROC AUC Score:", roc_auc_score(y_test, proba))
    return model

# Train models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
trained_rf = train_and_evaluate(rf, "Random Forest + SMOTE")

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
trained_gb = train_and_evaluate(gb, "Gradient Boosting + SMOTE")

# Save the better one
joblib.dump(trained_gb, '../data/readmission_model_smote.pkl')
print("\n✅ SMOTE-based model saved as 'readmission_model_smote.pkl'")
