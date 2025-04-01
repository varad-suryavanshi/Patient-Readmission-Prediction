import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

# Load preprocessed data
X_train = joblib.load('../data/X_train.pkl')
X_test = joblib.load('../data/X_test.pkl')
y_train = joblib.load('../data/y_train.pkl')
y_test = joblib.load('../data/y_test.pkl')

def train_and_evaluate(model, name):
    print(f"\n🔹 Training: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("ROC AUC Score:", roc_auc_score(y_test, proba))
    return model

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
trained_rf = train_and_evaluate(rf, "Random Forest")

# Train Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
trained_gb = train_and_evaluate(gb, "Gradient Boosting")

# Save best model (can compare AUC manually)
joblib.dump(trained_gb, '../models/readmission_model.pkl')
print("\n✅ Model saved as 'readmission_model.pkl'")
