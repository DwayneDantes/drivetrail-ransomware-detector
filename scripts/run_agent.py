import sqlite3
import pandas as pd
import time
import os
import joblib
import xgboost as xgb
import numpy as np
import json
from pathlib import Path
import sys

# --- Path Correction ---
# This is the standard way to make a script aware of the project's root directory.
# It adds the parent directory of 'scripts' (which is 'drivetrail') to the system path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
# --- End Path Correction ---


# Import our custom modules
from scripts.feature_extractor import extract_features
from scripts.bayes_fusion import fuse_evidence, detect_high_velocity_modification, detect_systematic_renaming

# --- Configuration ---
EVENT_DB_PATH = os.path.join('data', 'events.db')
XGB_MODEL_PATH = os.path.join('models', 'xgb_drivetrail.model')
HMM_MODEL_PATH = os.path.join('models', 'hmm_drivetrail.joblib')
AGENT_POLLING_INTERVAL = 3
EVENT_WINDOW_SECONDS = 60 # Time window for micropatterns (60 seconds)
HMM_SEQUENCE_LENGTH = 15 # Must match the length used for training

def load_models():
    """Loads all the trained models into memory."""
    print("Loading models...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    print("  - XGBoost model loaded.")
    hmm_model = joblib.load(HMM_MODEL_PATH)
    print("  - HMM model loaded.")
    return xgb_model, hmm_model

def read_new_events(conn, last_processed_id):
    """Reads all events from the database newer than the last processed ID."""
    query = f"SELECT * FROM events WHERE id > {last_processed_id} ORDER BY id ASC"
    df = pd.read_sql_query(query, conn)
    return df

def align_hmm_states(hmm_model):
    """Identifies which HMM state is 'malicious' based on mean entropy."""
    state_means = hmm_model.means_.flatten()
    malicious_state_index = np.argmax(state_means)
    print(f"HMM malicious state identified as State {malicious_state_index} (Mean Entropy: {state_means[malicious_state_index]:.2f})")
    return malicious_state_index

def main():
    """Main agent loop to process events and make predictions."""
    print("--- Starting DriveTrail Detection Agent ---")
    xgb_model, hmm_model = load_models()
    malicious_state_index = align_hmm_states(hmm_model)

    # State tracking variables
    last_processed_event_id = 0
    file_state_tracker = {}
    recent_events_df = pd.DataFrame()
    hmm_sequence = []

    print(f"Agent started. Polling every {AGENT_POLLING_INTERVAL} seconds. Press Ctrl+C to stop.")
    try:
        while True:
            with sqlite3.connect(EVENT_DB_PATH) as conn:
                new_events_df = read_new_events(conn, last_processed_event_id)
            
            if not new_events_df.empty:
                print(f"\n--- Found {len(new_events_df)} new event(s) to process ---")
                
                # Append new events to our rolling window for micropatterns
                recent_events_df = pd.concat([recent_events_df, new_events_df], ignore_index=True)
                
                for _, event in new_events_df.iterrows():
                    event_dict = event.to_dict()
                    print(f"Processing event ID {event_dict['id']}: {event_dict['event_type']} on '{event_dict['full_path']}'")

                    # 1. Extract Features
                    feature_vector, file_state_tracker = extract_features(event_dict, file_state_tracker)

                    # 2. Get XGBoost Score
                    xgb_prob = xgb_model.predict_proba(feature_vector)[0][1] # Probability of ransom
                    print(f"  - XGBoost Score: {xgb_prob:.4f}")

                    # 3. Detect Micropatterns
                    is_high_velocity = detect_high_velocity_modification(recent_events_df, window_seconds=EVENT_WINDOW_SECONDS)
                    is_systematic_rename = detect_systematic_renaming(recent_events_df, window_seconds=EVENT_WINDOW_SECONDS)
                    micropatterns = {'high_velocity': is_high_velocity, 'systematic_rename': is_systematic_rename}

                    # 4. Fuse Evidence
                    fused_prob = fuse_evidence(xgb_prob, micropatterns)
                    print(f"  - Fused Probability (after evidence): {fused_prob:.4f}")

                    # 5. Update HMM Sequence and Predict State
                    # The HMM was trained on entropy, so we use that as the observation
                    observation = feature_vector[['file_name_entropy']].values
                    hmm_sequence.append(observation)
                    
                    # Keep the sequence at a fixed length
                    if len(hmm_sequence) > HMM_SEQUENCE_LENGTH:
                        hmm_sequence.pop(0)

                    if len(hmm_sequence) >= HMM_SEQUENCE_LENGTH:
                        # Use the HMM to predict the sequence of hidden states
                        logprob, state_sequence = hmm_model.decode(np.vstack(hmm_sequence))
                        current_state = state_sequence[-1] # The state of the most recent event
                        
                        # Check for alert
                        if current_state == malicious_state_index:
                            print("\n" + "="*20 + " ALERT! " + "="*20)
                            print(f"  MALICIOUS SEQUENCE DETECTED! System has entered State {current_state}.")
                            print("="*50 + "\n")
                        else:
                            print(f"  - HMM State: Benign (State {current_state})")

                    last_processed_event_id = event_dict['id']

                # Prune the rolling event window to save memory
                cutoff_time = time.time() - EVENT_WINDOW_SECONDS
                recent_events_df = recent_events_df[recent_events_df['timestamp'] >= cutoff_time]
                print("--- Finished processing batch ---")

            time.sleep(AGENT_POLLING_INTERVAL)

    except KeyboardInterrupt:
        print("\n--- Agent stopped by user. ---")
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", e.__traceback__)

if __name__ == "__main__":
    main()