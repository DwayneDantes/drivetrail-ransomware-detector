import pandas as pd
import numpy as np
import os
from Levenshtein import ratio as levenshtein_ratio
import math

def calculate_shannon_entropy(s):
    """Calculates the Shannon entropy of a string."""
    if not s: return 0
    counts = pd.Series(list(s)).value_counts()
    probabilities = counts / len(s)
    return -np.sum(probabilities * np.log2(probabilities))

def extract_features(event, previous_event_state):
    """
    Converts a raw event dictionary (now with full_path) into a feature vector.
    """
    # --- Get core event data ---
    event_type = event.get('event_type', 'UNKNOWN')
    local_title = event.get('local_title', '')
    stable_id = event.get('stable_id')
    full_path = event.get('full_path', '') # Get the new full_path

    # --- NEW: Calculate path-based features ---
    path_length = len(full_path)
    directory_depth = full_path.count(os.sep)
    
    # Check for suspicious keywords in the path
    suspicious_keywords = ['appdata', 'temp', 'tmp', 'programdata', 'userprofile']
    is_suspicious_path = 1 if any(keyword in full_path.lower() for keyword in suspicious_keywords) else 0

    # Get the last known title for this specific file to calculate extension similarity
    last_known_title = previous_event_state.get(stable_id, {}).get('local_title', '')

    # --- Initialize and Calculate All 12 Features ---
    features = {
        'File_Delete_archived': 1 if event_type in ['DELETED', 'TRASHED'] else 0,
        'File_created': 1 if event_type == 'CREATED' else 0,
        'process-related': 1 if any(local_title.lower().endswith(ext) for ext in ['.exe', '.bat', '.ps1']) else 0,
        'network-related': 1, # Proxy: Assume all Drive events are network-related
        'file-related': 1,
        'suspicious_path': is_suspicious_path,
        'system_executable': 1 if any(local_title.lower().endswith(ext) for ext in ['.exe', '.dll']) else 0,
        'path_length': path_length,
        'directory_depth': directory_depth,
        'process_name_length': len(local_title),
        'extension_similarity': levenshtein_ratio(os.path.splitext(last_known_title)[1], os.path.splitext(local_title)[1]) if event_type == 'RENAMED' and last_known_title else 1.0,
        'file_name_entropy': calculate_shannon_entropy(local_title)
    }

    # Update the state dictionary with the new title for this file
    if stable_id:
        previous_event_state[stable_id] = {'local_title': local_title}

    # Ensure the DataFrame has columns in the exact order the model expects
    feature_order = [
        'File_Delete_archived', 'File_created', 'process-related', 'network-related',
        'file-related', 'suspicious_path', 'system_executable', 'path_length',
        'directory_depth', 'process_name_length', 'extension_similarity', 'file_name_entropy'
    ]
    
    return pd.DataFrame([features], columns=feature_order), previous_event_state

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Testing Upgraded Feature Extractor ---")
    
    file_state_tracker = {}
    
    # Simulate a RENAME event with a full path
    rename_event = {
        'timestamp': 1760918005.0, 'stable_id': 1234, 'event_type': 'RENAMED',
        'local_title': 'mydocument.encrypted', 'old_local_title': 'mydocument.docx',
        'full_path': 'MyProject\\SubFolder\\mydocument.encrypted'
    }
    # We need to prime the state tracker with the old name
    file_state_tracker[1234] = {'local_title': 'mydocument.docx'}

    features_df, file_state_tracker = extract_features(rename_event, file_state_tracker)
    print("\nFeatures for RENAME event with path:")
    print(features_df.to_string())

    print(f"\nPath Length: {features_df['path_length'].iloc[0]} (Expected > 0)")
    print(f"Directory Depth: {features_df['directory_depth'].iloc[0]} (Expected: 2)")
