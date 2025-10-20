import pandas as pd
import xgboost as xgb
import shap
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
INPUT_DATA_PATH = os.path.join( 'data', 'training_features.parquet')
MODEL_OUTPUT_PATH = os.path.join( 'models', 'xgb_drivetrail.model')

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
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Model Training ---
    print("Training the XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=42, use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
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
    explainer = shap.TreeExplainer(model.get_booster(), X_train)
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