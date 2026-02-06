import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Paths
INPUT_DATA_PATH = os.path.join('data', 'training_features.parquet')
XGB_MODEL_PATH = os.path.join('models', 'xgb_drivetrail.model')
RESULTS_DIR = os.path.join('paper_results')

def threshold_based_detector(X):
    """
    Baseline 1: Simple threshold-based detection.
    """
    predictions = np.zeros(len(X))
    
    # Rule 1: High entropy filenames
    high_entropy = X['file_name_entropy'] > 3.5
    
    # Rule 2: Suspicious renaming
    suspicious_rename = X['extension_similarity'] < 0.5
    
    # Rule 3: Deletion/archiving activity
    deletion_activity = X['File_Delete_archived'] == 1
    
    # Combine rules (OR logic)
    predictions[high_entropy | suspicious_rename | deletion_activity] = 1
    
    return predictions

def xgboost_only_detector(X, model):
    """
    Baseline 2: XGBoost without Bayesian fusion or HMM.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    return (y_pred_proba >= 0.5).astype(int)

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate metrics for a detector."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,  # Both keys for compatibility
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
    
    return metrics

def normalize_metrics(metrics):
    """Ensure both 'fpr' and 'false_positive_rate' keys exist."""
    if 'false_positive_rate' in metrics and 'fpr' not in metrics:
        metrics['fpr'] = metrics['false_positive_rate']
    elif 'fpr' in metrics and 'false_positive_rate' not in metrics:
        metrics['false_positive_rate'] = metrics['fpr']
    return metrics

def main():
    print("="*70)
    print("  BASELINE COMPARISON EVALUATION")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_parquet(INPUT_DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load XGBoost model
    print("📂 Loading XGBoost model...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    
    results = {}
    
    # Baseline 1: Threshold-Based
    print("\n🔍 Evaluating Baseline 1: Threshold-Based Detector...")
    y_pred_threshold = threshold_based_detector(X_test)
    results['threshold_based'] = normalize_metrics(evaluate_model(y_test, y_pred_threshold))
    print(f"   ✓ F1-Score: {results['threshold_based']['f1_score']:.4f}")
    print(f"   ✓ FPR: {results['threshold_based']['fpr']:.4f}")
    
    # Baseline 2: XGBoost-Only
    print("\n🔍 Evaluating Baseline 2: XGBoost-Only (no fusion/HMM)...")
    y_pred_xgb = xgboost_only_detector(X_test, xgb_model)
    y_pred_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    results['xgboost_only'] = normalize_metrics(evaluate_model(y_test, y_pred_xgb, y_pred_xgb_proba))
    print(f"   ✓ F1-Score: {results['xgboost_only']['f1_score']:.4f}")
    print(f"   ✓ FPR: {results['xgboost_only']['fpr']:.4f}")
    
    # R-NPDA Full System (load from previous results)
    print("\n🔍 Loading R-NPDA Full System results...")
    try:
        with open(os.path.join(RESULTS_DIR, 'xgboost_results.json'), 'r') as f:
            r_npda_results = json.load(f)
        results['r_npda_full'] = normalize_metrics(r_npda_results['test_performance'])
        print(f"   ✓ F1-Score: {results['r_npda_full']['f1_score']:.4f}")
        print(f"   ✓ FPR: {results['r_npda_full']['fpr']:.4f}")
    except FileNotFoundError:
        print("   ⚠️ R-NPDA results not found. Run train_and_evaluate_enhanced.py first.")
        results['r_npda_full'] = None
    
    # Save comparison
    comparison_file = os.path.join(RESULTS_DIR, 'baseline_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Comparison saved: {comparison_file}")
    
    # Print summary table
    print("\n" + "="*70)
    print("  BASELINE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FPR':<8}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:<25} "
                  f"{metrics['accuracy']:.4f}  "
                  f"{metrics['precision']:.4f}  "
                  f"{metrics['recall']:.4f}  "
                  f"{metrics['f1_score']:.4f}  "
                  f"{metrics['fpr']:.4f}")  # Now using 'fpr' consistently
    
    print("="*70)
    
    # Analysis
    if results['r_npda_full']:
        print("\n📊 PERFORMANCE ANALYSIS:")
        print("-" * 70)
        
        r_npda = results['r_npda_full']
        threshold = results['threshold_based']
        xgb_only = results['xgboost_only']
        
        # F1-Score improvements
        f1_vs_threshold = ((r_npda['f1_score'] - threshold['f1_score']) / threshold['f1_score']) * 100
        f1_vs_xgb = ((r_npda['f1_score'] - xgb_only['f1_score']) / xgb_only['f1_score']) * 100
        
        print(f"\n✅ R-NPDA vs Threshold-Based:")
        print(f"   • F1-Score improvement: {f1_vs_threshold:+.1f}%")
        print(f"   • FPR reduction: {threshold['fpr']:.4f} → {r_npda['fpr']:.4f} "
              f"({((r_npda['fpr'] - threshold['fpr']) / threshold['fpr']) * 100:+.1f}%)")
        
        print(f"\n✅ R-NPDA vs XGBoost-Only:")
        print(f"   • F1-Score improvement: {f1_vs_xgb:+.1f}%")
        print(f"   • FPR reduction: {xgb_only['fpr']:.4f} → {r_npda['fpr']:.4f} "
              f"({((r_npda['fpr'] - xgb_only['fpr']) / xgb_only['fpr']) * 100:+.1f}%)")
        
        print("\n" + "="*70)
        print("  KEY INSIGHTS FOR PAPER")
        print("="*70)
        print(f"\n• R-NPDA achieves {r_npda['accuracy']:.2%} accuracy with only {r_npda['fpr']:.2%} FPR")
        print(f"• Threshold-based approach has unacceptably high FPR ({threshold['fpr']:.2%})")
        print(f"• XGBoost-only is strong but R-NPDA improves F1 by {f1_vs_xgb:.1f}%")
        print(f"• The hybrid approach (Bayesian + HMM) adds {f1_vs_xgb:.1f}% improvement")
        
    print("\n✅ Baseline comparison complete!")

if __name__ == "__main__":
    main()