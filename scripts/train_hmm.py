import pandas as pd
import numpy as np
import os
import joblib
from hmmlearn.hmm import GaussianHMM

# --- Configuration ---
INPUT_DATA_PATH = os.path.join('data', 'training_features.parquet')
MODEL_OUTPUT_PATH = os.path.join('models', 'hmm_drivetrail.joblib') # Use .joblib for scikit-learn models
SEQUENCE_LENGTH = 15
NUM_SEQUENCES_PER_CLASS = 500
N_STATES = 2 # The number of hidden states (Benign, Malicious)

def create_sequences(df, n_sequences, seq_length):
    """Helper function to create sequences from the dataframe."""
    sequences = []
    for _ in range(n_sequences):
        start_index = np.random.randint(0, len(df) - seq_length)
        sequence = df.iloc[start_index:start_index + seq_length].values
        sequences.append(sequence)
    return sequences

def main():
    """
    Main function to train and save the HMM model using hmmlearn.
    """
    print("--- Starting HMM Model Training Script (hmmlearn) ---")

    # --- Load Data ---
    print(f"Loading data from '{INPUT_DATA_PATH}'...")
    df = pd.read_parquet(INPUT_DATA_PATH, columns=['label', 'file_name_entropy'])
    benign_df = df[df['label'] == 0].drop('label', axis=1)
    ransom_df = df[df['label'] == 1].drop('label', axis=1)
    print(f"Loaded {len(benign_df)} benign and {len(ransom_df)} malicious samples.")

    # --- Create Training Sequences ---
    print(f"Generating {NUM_SEQUENCES_PER_CLASS} sequences of length {SEQUENCE_LENGTH} for each class...")
    benign_sequences = create_sequences(benign_df, NUM_SEQUENCES_PER_CLASS, SEQUENCE_LENGTH)
    ransom_sequences = create_sequences(ransom_df, NUM_SEQUENCES_PER_CLASS, SEQUENCE_LENGTH)
    all_sequences = benign_sequences + ransom_sequences

    # --- Prepare Data for hmmlearn ---
    # hmmlearn's .fit() method expects a 2D numpy array of all observations concatenated together,
    # and a separate array describing the length of each sequence.
    X_concatenated = np.concatenate(all_sequences)
    lengths = [len(seq) for seq in all_sequences]

    # --- Define and Train HMM Structure ---
    print(f"Defining and training GaussianHMM with {N_STATES} states...")
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag", # Each feature has its own variance
        n_iter=100,            # Number of iterations for the learning algorithm
        random_state=42,
        verbose=True
    )

    # Fit the model to the data
    model.fit(X_concatenated, lengths)
    print("\nModel training complete.")

    # --- Align States (Interpretability) ---
    # We find the state with the higher mean file_name_entropy and label it "malicious"
    state_means = model.means_.flatten() # Flatten to make it a simple 1D array
    malicious_state_index = np.argmax(state_means)
    benign_state_index = np.argmin(state_means)

    print(f"\nDiscovered States (based on mean file_name_entropy):")
    print(f"  - Benign State (ID {benign_state_index}): Mean = {state_means[benign_state_index]:.4f}")
    print(f"  - Malicious State (ID {malicious_state_index}): Mean = {state_means[malicious_state_index]:.4f}")

    # --- Save Model ---
    print(f"\nSaving the trained model to '{MODEL_OUTPUT_PATH}'...")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")

    print("\n--- HMM Script finished successfully! ---")

if __name__ == "__main__":
    main()