import pandas as pd
import os

# --- Configuration ---
# Define the columns we want to keep for our Phase 1 model.
COLUMNS_TO_KEEP = [
    'Ware Type',
    'File_Delete_archived',
    'File_created',
    'process-related',
    'network-related',
    'file-related',
    'suspicious_path',
    'system_executable',
    'path_length',
    'directory_depth',
    'process_name_length',
    'extension_similarity',
    'file_name_entropy'
]

# --- UPDATED FILE PATHS ---
# This now points to your specific CSV file.
INPUT_FILE_PATH = os.path.join('data', 'raw', 'Ransomware_Data.csv')
OUTPUT_FILE_PATH = os.path.join('data', 'training_features.parquet')

def main():
    """
    Main function to load, process, and save the dataset.
    """
    print(f"--- Starting Data Synthesis Script ---")

    # Check if the input file exists
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"ERROR: Input file not found at '{INPUT_FILE_PATH}'")
        print("Please ensure your 'Ransomware_Data.csv' file is in the 'data/raw' folder.")
        return

    # --- UPDATED DATA LOADING ---
    # Load the raw dataset. We now use sep=',' for a CSV file.
    print(f"Loading raw data from '{INPUT_FILE_PATH}'...")
    try:
        df = pd.read_csv(INPUT_FILE_PATH, sep=',')
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"ERROR: Could not read the CSV file. Please check its format. Details: {e}")
        return

    # --- Feature Selection ---
    print(f"Selecting the {len(COLUMNS_TO_KEEP)} relevant columns for Phase 1...")
    df_processed = df[COLUMNS_TO_KEEP].copy()

    # --- Label Encoding ---
    # Convert the 'Ware Type' column from text ('good', 'ransom') to numbers (0, 1).
    print("Encoding 'Ware Type' label (good=0, ransom=1)...")
    df_processed['Ware Type'] = df_processed['Ware Type'].map({'good': 0, 'ransom': 1})

    # Rename the column to 'label' for clarity in the model.
    df_processed.rename(columns={'Ware Type': 'label'}, inplace=True)

    # --- Final Validation and Save ---
    print(f"Processed data has {df_processed.shape[0]} rows and {df_processed.shape[1]} columns.")
    print("Final columns:", df_processed.columns.tolist())
    print("\nSample of the processed data:")
    print(df_processed.head())

    # Save the final, clean DataFrame to a Parquet file for efficient use later.
    print(f"\nSaving processed data to '{OUTPUT_FILE_PATH}'...")
    df_processed.to_parquet(OUTPUT_FILE_PATH, index=False)

    print(f"--- Script finished successfully! ---")
    print(f"Your training-ready data is now available at '{OUTPUT_FILE_PATH}'")


if __name__ == "__main__":
    main()