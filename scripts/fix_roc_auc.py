import json
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RESULTS_DIR = 'paper_results'
INPUT_DATA_PATH = os.path.join('data', 'training_features.parquet')
XGB_MODEL_PATH = os.path.join('models', 'xgb_drivetrail.model')

print("🔧 Fixing ROC-AUC scores...")

# Load data
df = pd.read_parquet(INPUT_DATA_PATH)
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

# Get probabilities
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = float(roc_auc_score(y_test, y_pred_proba))

print(f"✅ Calculated ROC-AUC: {roc_auc:.4f}")

# Update xgboost_results.json
results_file = os.path.join(RESULTS_DIR, 'xgboost_results.json')
with open(results_file, 'r') as f:
    results = json.load(f)

results['test_performance']['roc_auc'] = roc_auc

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Updated: {results_file}")

# Update baseline_comparison.json
comparison_file = os.path.join(RESULTS_DIR, 'baseline_comparison.json')
if os.path.exists(comparison_file):
    with open(comparison_file, 'r') as f:
        comparison = json.load(f)
    
    # Add ROC-AUC to xgboost_only
    comparison['xgboost_only']['roc_auc'] = roc_auc
    
    # Add to r_npda_full (same model, same data)
    if comparison.get('r_npda_full'):
        comparison['r_npda_full']['roc_auc'] = roc_auc
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"✅ Updated: {comparison_file}")

print("\n📊 Final ROC-AUC values:")
print(f"   • Threshold-Based: N/A (no probability scores)")
print(f"   • XGBoost-Only: {roc_auc:.4f}")
print(f"   • R-NPDA Full: {roc_auc:.4f}")
print("\n✅ All fixed!")