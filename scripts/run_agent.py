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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Import our custom modules
from scripts.feature_extractor import extract_features
from scripts.bayes_fusion import fuse_evidence, detect_high_velocity_modification, detect_systematic_renaming
from scripts.response_actions import DriveResponseSystem

# --- Configuration ---
EVENT_DB_PATH = os.path.join('data', 'events.db')
XGB_MODEL_PATH = os.path.join('models', 'xgb_drivetrail.model')
HMM_MODEL_PATH = os.path.join('models', 'hmm_drivetrail.joblib')
AGENT_POLLING_INTERVAL = 3
EVENT_WINDOW_SECONDS = 60
HMM_SEQUENCE_LENGTH = 15

# --- NEW: Alerting Configuration ---
ALERT_THRESHOLD = 0.50  # Minimum probability to trigger any alert
PAUSE_THRESHOLD = 0.75  # Minimum probability to pause sync
CRITICAL_THRESHOLD = 0.90  # Emergency threshold

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

def count_recent_files_affected(events_df, window_seconds=60):
    """Counts how many unique files were affected in recent window."""
    if events_df.empty:
        return 0
    current_time = events_df['timestamp'].max()
    window_start = current_time - window_seconds
    recent = events_df[events_df['timestamp'] >= window_start]
    return recent['stable_id'].nunique()

def main():
    """Main agent loop to process events and make predictions."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          DriveTrail Detection Agent - ACTIVE                ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    xgb_model, hmm_model = load_models()
    malicious_state_index = align_hmm_states(hmm_model)
    
    # Initialize response system
    response_system = DriveResponseSystem()

    # State tracking variables
    last_processed_event_id = 0
    file_state_tracker = {}
    recent_events_df = pd.DataFrame()
    hmm_sequence = []
    
    # Track if we're in alert mode (to avoid spam)
    alert_cooldown = 0
    COOLDOWN_SECONDS = 30  # Don't spam alerts more than once per 30 seconds

    print(f"Agent started. Polling every {AGENT_POLLING_INTERVAL} seconds.")
    print(f"Alert threshold: {ALERT_THRESHOLD:.1%} | Pause threshold: {PAUSE_THRESHOLD:.1%}")
    print(f"Press Ctrl+C to stop.\n")
    
    try:
        while True:
            current_time = time.time()
            
            # Check if we're still in cooldown
            if alert_cooldown > current_time:
                time.sleep(AGENT_POLLING_INTERVAL)
                continue
            
            with sqlite3.connect(EVENT_DB_PATH) as conn:
                new_events_df = read_new_events(conn, last_processed_event_id)
            
            if not new_events_df.empty:
                print(f"\n{'─'*60}")
                print(f"[{time.strftime('%H:%M:%S')}] Processing {len(new_events_df)} new event(s)...")
                
                # Append new events to our rolling window
                recent_events_df = pd.concat([recent_events_df, new_events_df], ignore_index=True)
                
                # Track maximum threat in this batch
                max_threat_prob = 0.0
                max_threat_event = None
                
                for _, event in new_events_df.iterrows():
                    event_dict = event.to_dict()
                    
                    # 1. Extract Features
                    feature_vector, file_state_tracker = extract_features(event_dict, file_state_tracker)

                    # 2. Get XGBoost Score
                    xgb_prob = xgb_model.predict_proba(feature_vector)[0][1]

                    # 3. Detect Micropatterns
                    is_high_velocity = detect_high_velocity_modification(recent_events_df, window_seconds=EVENT_WINDOW_SECONDS)
                    is_systematic_rename = detect_systematic_renaming(recent_events_df, window_seconds=EVENT_WINDOW_SECONDS)
                    micropatterns = {'high_velocity': is_high_velocity, 'systematic_rename': is_systematic_rename}

                    # 4. Fuse Evidence
                    fused_prob = fuse_evidence(xgb_prob, micropatterns)
                    
                    # Display event summary
                    event_type = event_dict.get('event_type', 'UNKNOWN')
                    file_path = event_dict.get('full_path', event_dict.get('local_title', 'unknown'))
                    print(f"  [{event_type}] {file_path}")
                    print(f"    XGB: {xgb_prob:.3f} → Fused: {fused_prob:.3f}", end='')
                    
                    # Visual indicator
                    if fused_prob >= CRITICAL_THRESHOLD:
                        print(" 🚨 CRITICAL")
                    elif fused_prob >= PAUSE_THRESHOLD:
                        print(" 🟠 HIGH")
                    elif fused_prob >= ALERT_THRESHOLD:
                        print(" 🟡 MEDIUM")
                    else:
                        print(" ✓")

                    # Track highest threat in batch
                    if fused_prob > max_threat_prob:
                        max_threat_prob = fused_prob
                        max_threat_event = event_dict

                    # 5. Update HMM Sequence
                    observation = feature_vector[['file_name_entropy']].values
                    hmm_sequence.append(observation)
                    
                    if len(hmm_sequence) > HMM_SEQUENCE_LENGTH:
                        hmm_sequence.pop(0)

                    if len(hmm_sequence) >= HMM_SEQUENCE_LENGTH:
                        logprob, state_sequence = hmm_model.decode(np.vstack(hmm_sequence))
                        current_state = state_sequence[-1]
                        
                        if current_state == malicious_state_index:
                            print(f"    ⚠️  HMM: Malicious sequence detected (State {current_state})")
                            # Boost the threat probability if HMM confirms
                            max_threat_prob = max(max_threat_prob, 0.85)

                    last_processed_event_id = event_dict['id']

                # --- RESPONSE DECISION ---
                # After processing batch, decide if action is needed
                if max_threat_prob >= ALERT_THRESHOLD:
                    threat_level = response_system.get_threat_level(max_threat_prob)
                    affected_count = count_recent_files_affected(recent_events_df)
                    
                    print(f"\n{'═'*60}")
                    print(f"  ⚠️  THREAT DETECTED: {threat_level} ({max_threat_prob:.1%})")
                    print(f"  Affected files in window: {affected_count}")
                    print(f"{'═'*60}")
                    
                    # Take action through response system
                    response_system.take_action(
                        threat_level=threat_level,
                        probability=max_threat_prob,
                        event_details=max_threat_event,
                        affected_files_count=affected_count
                    )
                    
                    # Set cooldown to avoid spam
                    alert_cooldown = current_time + COOLDOWN_SECONDS
                    
                    # If critical, generate recovery report
                    if threat_level == 'CRITICAL':
                        response_system.generate_recovery_report()
                        print("\n📄 Recovery instructions have been generated.")
                        print("   See: data/recovery_instructions.txt\n")

                # Prune the rolling event window
                cutoff_time = current_time - EVENT_WINDOW_SECONDS
                recent_events_df = recent_events_df[recent_events_df['timestamp'] >= cutoff_time]

            time.sleep(AGENT_POLLING_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n{'═'*60}")
        print("  Agent stopped by user.")
        print(f"{'═'*60}\n")
        
        # Cleanup: if sync was paused, remind user
        if response_system.sync_paused:
            print("⚠️  REMINDER: Drive sync is currently PAUSED.")
            print("   Run 'python scripts/resume_sync.py' when safe to resume.\n")
            
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency pause if we crash during an alert
        if response_system and not response_system.sync_paused:
            print("\n⚠️  Emergency: Pausing Drive sync due to agent crash...")
            response_system.pause_drive_sync(method='suspend')

if __name__ == "__main__":
    main()