import pandas as pd
import xgboost as xgb
import shap
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
INPUT_DATA_PATH = os.path.join( 'data', 'training_features.parquet')
MODEL_OUTPUT_PATH = os.path.join( 'models', 'xgb_drivetrail.model')
RANDOM_STATE = 42

def main():
    """
    Main function to train, evaluate, and save the XGBoost model.
    """
    print("--- Starting Full Training & Evaluation Script ---")

    # --- Load Data ---
    print(f"Loading data from '{INPUT_DATA_PATH}'...")
    df = pd.read_parquet(INPUT_DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']

    # --- Train/Test Split ---
    print("Splitting data into train/validation/test sets (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    # --- Model Training ---
    print("Running randomized search with stratified 5-fold CV...")
    scale_pos_weight = (y_train == 'good').sum() / max((y_train == 'ransom').sum(), 1)
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        use_label_encoder=False
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
    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    print("Training final model on train+validation set...")
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    model = search.best_estimator_
    model.fit(X_train_full, y_train_full)
    print("Model training complete.")

    # --- Performance Evaluation ---
    print("\n--- Model Performance Report ---")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['good', 'ransom']))

    # --- Confusion Matrix Visualization ---
    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['good', 'ransom']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # --- SHAP Feature Importance Analysis ---
    print("\n--- SHAP Feature Importance ---")
    print("Calculating SHAP values... (this may take a moment)")
    explainer = shap.TreeExplainer(model.get_booster(), X_train_full)
    X_test_sample = X_test.sample(n=1000, random_state=42) # Use a sample for speed
    shap_values = explainer.shap_values(X_test_sample)

    print("Generating SHAP summary plot (bar chart)...")
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP)")
    plt.show()
    
    print("Generating SHAP summary plot (detailed dot plot)...")
    shap.summary_plot(shap_values, X_test_sample, show=True)


    # --- Save Model (at the very end) ---
    print(f"\nSaving the trained model to '{MODEL_OUTPUT_PATH}'...")
    model.save_model(MODEL_OUTPUT_PATH)
    print("Model saved successfully.")

    print("\n--- Script finished successfully! ---")

if __name__ == "__main__":
    main()
