import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

# --- Configuration ---
INPUT_DATA_PATH = os.path.join('data', 'training_features.parquet')
MODEL_OUTPUT_PATH = os.path.join('models', 'xgb_drivetrail.model')
RESULTS_DIR = os.path.join('paper_results')
RANDOM_STATE = 42

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)


def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics for paper."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positive_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'average_precision': float(average_precision_score(y_true, y_pred_proba)),
        'total_samples': len(y_true),
        'positive_samples': int(np.sum(y_true)),
        'negative_samples': int(len(y_true) - np.sum(y_true))
    }

    return metrics


def plot_confusion_matrix(cm, save_path):
    """Generate and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign', 'Ransomware'],
        yticklabels=['Benign', 'Ransomware'],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - XGBoost Classifier', fontsize=14, fontweight='bold')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j + 0.5,
                i + 0.7,
                f'({cm_normalized[i, j]:.1%})',
                ha='center',
                va='center',
                fontsize=10,
                color='gray'
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Confusion matrix saved: {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Generate and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ ROC curve saved: {save_path}")


def plot_precision_recall_curve(precision, recall, avg_precision, save_path):
    """Generate and save Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ PR curve saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=12):
    """Generate and save feature importance plot."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Feature importance saved: {save_path}")


def summarize_cv_scores(scores):
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'scores': scores.tolist()
    }


def main():
    """Main training and evaluation workflow."""
    print("=" * 70)
    print("  DRIVETRAIL ENHANCED TRAINING & EVALUATION")
    print("  (Comprehensive Metrics for IEEE Paper)")
    print("=" * 70)

    # --- Load Data ---
    print(f"\n📂 Loading data from '{INPUT_DATA_PATH}'...")
    df = pd.read_parquet(INPUT_DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']
    feature_names = X.columns.tolist()

    print(f"   ✓ Loaded {len(df)} samples")
    print(f"   ✓ Benign: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"   ✓ Ransomware: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    print(f"   ✓ Features: {len(feature_names)}")

    # --- Train/Validation/Test Split ---
    print("\n📊 Splitting data (70% train, 15% validation, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"   ✓ Train: {len(X_train)} samples")
    print(f"   ✓ Validation: {len(X_val)} samples")
    print(f"   ✓ Test: {len(X_test)} samples")

    # --- Hyperparameter Search ---
    print("\n🔎 Running randomized search with stratified 5-fold CV...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=1
    )
    param_distributions = {
        'n_estimators': np.arange(100, 1001, 100),
        'max_depth': np.arange(3, 10),
        'learning_rate': np.linspace(0.01, 0.3, 10),
        'subsample': np.linspace(0.6, 1.0, 5),
        'colsample_bytree': np.linspace(0.6, 1.0, 5),
        'scale_pos_weight': [scale_pos_weight]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=25,
        scoring='f1',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    print(f"   ✓ Best params: {search.best_params_}")
    print(f"   ✓ Best CV F1: {search.best_score_:.4f}")

    # --- Train Final Model ---
    print("\n🤖 Training final model on train+validation sets...")
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    model = search.best_estimator_
    model.fit(X_train_full, y_train_full)
    print("   ✓ Model training complete")

    # --- Test Set Evaluation ---
    print("\n📈 Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)

    print("\n" + "=" * 70)
    print("  TEST SET PERFORMANCE")
    print("=" * 70)
    print(f"  Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  Precision:  {test_metrics['precision']:.4f}")
    print(f"  Recall:     {test_metrics['recall']:.4f}")
    print(f"  F1-Score:   {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    print(f"  FPR:        {test_metrics['false_positive_rate']:.4f}")
    print("=" * 70)

    # --- Generate Visualizations ---
    print("\n📊 Generating visualizations...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, 'figures', 'confusion_matrix.png'))

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    plot_roc_curve(fpr, tpr, test_metrics['roc_auc'], os.path.join(RESULTS_DIR, 'figures', 'roc_curve.png'))

    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    plot_precision_recall_curve(
        precision,
        recall,
        test_metrics['average_precision'],
        os.path.join(RESULTS_DIR, 'figures', 'precision_recall_curve.png')
    )

    plot_feature_importance(
        model,
        feature_names,
        os.path.join(RESULTS_DIR, 'figures', 'feature_importance.png')
    )

    # --- SHAP Feature Importance Analysis ---
    print("\n--- SHAP Feature Importance ---")
    print("Calculating SHAP values... (this may take a moment)")
    explainer = shap.TreeExplainer(model.get_booster(), X_train_full)
    X_test_sample = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
    shap_values = explainer.shap_values(X_test_sample)

    print("Generating SHAP summary plot (bar chart)...")
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP)")
    plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Generating SHAP summary plot (detailed dot plot)...")
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'shap_summary_dot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Save Comprehensive Results ---
    print("\n💾 Saving results...")
    results_package = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'XGBoost',
            'dataset': INPUT_DATA_PATH,
            'total_samples': len(df),
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'features': feature_names
        },
        'test_performance': test_metrics,
        'cross_validation': {
            'best_params': search.best_params_,
            'best_cv_f1': float(search.best_score_)
        },
        'roc_curve_data': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist() if roc_thresholds is not None else []
        },
        'pr_curve_data': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist() if pr_thresholds is not None else []
        },
        'classification_report': classification_report(
            y_test,
            y_pred,
            target_names=['Benign', 'Ransomware'],
            output_dict=True
        )
    }

    results_file = os.path.join(RESULTS_DIR, 'xgboost_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_package, f, indent=2)
    print(f"   ✓ Results saved: {results_file}")

    cm_data = {
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }
    cm_file = os.path.join(RESULTS_DIR, 'confusion_matrix_data.json')
    with open(cm_file, 'w') as f:
        json.dump(cm_data, f, indent=2)
    print(f"   ✓ Confusion matrix data saved: {cm_file}")

    # --- Save Model ---
    print(f"\n💾 Saving trained model...")
    model.save_model(MODEL_OUTPUT_PATH)
    print(f"   ✓ Model saved: {MODEL_OUTPUT_PATH}")

    # --- Summary for Paper ---
    print("\n" + "=" * 70)
    print("  SUMMARY FOR IEEE PAPER")
    print("=" * 70)
    print("\nTable I: Detection Performance Metrics")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 40)
    print(f"{'Accuracy':<20} {test_metrics['accuracy']:.4f}")
    print(f"{'Precision':<20} {test_metrics['precision']:.4f}")
    print(f"{'Recall':<20} {test_metrics['recall']:.4f}")
    print(f"{'F1-Score':<20} {test_metrics['f1_score']:.4f}")
    print(f"{'ROC-AUC':<20} {test_metrics['roc_auc']:.4f}")
    print(f"{'FPR':<20} {test_metrics['false_positive_rate']:.4f}")
    print("=" * 40)

    print("\n✅ Training and evaluation complete!")
    print(f"📁 All results saved to: {RESULTS_DIR}/")
    print(f"📊 Figures saved to: {RESULTS_DIR}/figures/")


if __name__ == "__main__":
    main()
