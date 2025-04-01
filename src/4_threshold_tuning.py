import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score,
    classification_report, confusion_matrix
)

# Load model, test data, and scaler (optional)
xgb = joblib.load('../data/xgb_engineered.pkl')
X_test = joblib.load('../data/df_engineered.pkl').drop('readmitted', axis=1)
y_test = joblib.load('../data/df_engineered.pkl')['readmitted']

# Use final train/test split (can reload from earlier too)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.2, random_state=42)

# Get predicted probabilities
y_scores = xgb.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

# PR Curve
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_scores)

# Plot ROC and PR Curves
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(thresholds_pr, precisions[:-1], label="Precision")
plt.plot(thresholds_pr, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.title("Precision-Recall vs Threshold")
plt.legend()

plt.tight_layout()
plt.savefig("../data/threshold_tuning_curves.png")
plt.show()

# Try threshold = 0.3 (as an example)
threshold = 0.3
y_pred_thresh = (y_scores >= threshold).astype(int)

print(f"\n🔍 Threshold set to: {threshold}")
print("Classification Report:")
print(classification_report(y_test, y_pred_thresh))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_thresh))

from sklearn.metrics import f1_score

# Try all thresholds to find best F1
best_thresh = 0.5
best_f1 = 0

for t in np.arange(0.05, 0.95, 0.01):
    y_pred = (y_scores >= t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n🚀 Best threshold by F1 score: {best_thresh:.2f} (F1 = {best_f1:.3f})")

# Show final metrics at best threshold
y_final = (y_scores >= best_thresh).astype(int)
print("\n📊 Classification Report at Best Threshold:")
print(classification_report(y_test, y_final))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_final))

