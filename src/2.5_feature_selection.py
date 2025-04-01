import joblib
import pandas as pd
import numpy as np

# Load full training data and model
X_train = joblib.load('../data/X_train.pkl')
X_test = joblib.load('../data/X_test.pkl')
y_train = joblib.load('../data/y_train.pkl')
y_test = joblib.load('../data/y_test.pkl')

# Clean feature names (match how XGBoost sees them)
X_train.columns = X_train.columns.str.replace('[<>\[\]]', '', regex=True)
X_test.columns = X_test.columns.str.replace('[<>\[\]]', '', regex=True)

# Load trained model
model = joblib.load('../data/readmission_model_xgboost.pkl')

# Get feature importance from model
booster = model.get_booster()
feature_scores = booster.get_score(importance_type='gain')

importance_df = pd.DataFrame({
    'feature': list(feature_scores.keys()),
    'importance': list(feature_scores.values())
}).sort_values(by='importance', ascending=False)

# Select top N features
TOP_N = 100
top_features = importance_df['feature'].head(TOP_N).tolist()

print(f"✅ Selected Top {TOP_N} Features")

# Reduce datasets
X_train_reduced = X_train[top_features]
X_test_reduced = X_test[top_features]

# Save reduced datasets
joblib.dump(X_train_reduced, '../data/X_train_reduced.pkl')
joblib.dump(X_test_reduced, '../data/X_test_reduced.pkl')
joblib.dump(y_train, '../data/y_train.pkl')
joblib.dump(y_test, '../data/y_test.pkl')

# Save feature importance for review
importance_df.to_csv('../data/feature_importance.csv', index=False)

print(f"\n✅ Saved reduced datasets and importance list.")
