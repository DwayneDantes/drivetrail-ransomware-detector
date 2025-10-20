# scripts/bayes_fusion.py
import pandas as pd
import time

# --- Likelihood Ratios (Configurable) ---
# These represent how much to multiply the odds by when a pattern is detected.
# A value of 3 means "this evidence makes ransomware 3 times more likely".
LIKELIHOOD_RATIOS = {
    'high_velocity': 3.0,
    'systematic_rename': 4.0,
}
# A small value to prevent division by zero in odds calculations
EPSILON = 1e-9

def detect_high_velocity_modification(events_df, window_seconds=60, threshold=50):
    """Detects if a large number of files have been modified in a short time window."""
    if events_df.empty:
        return False
    current_time = events_df['timestamp'].max()
    window_start_time = current_time - window_seconds
    recent_modifications = events_df[
        (events_df['timestamp'] >= window_start_time) &
        (events_df['event_type'].isin(['MODIFIED', 'CREATED', 'RENAMED']))
    ]
    return len(recent_modifications) > threshold

def detect_systematic_renaming(events_df, window_seconds=60, threshold=5):
    """Detects if multiple files have been renamed with a similar new extension."""
    if events_df.empty:
        return False
    current_time = events_df['timestamp'].max()
    window_start_time = current_time - window_seconds
    recent_renames = events_df[
        (events_df['timestamp'] >= window_start_time) &
        (events_df['event_type'] == 'RENAMED')
    ]
    if len(recent_renames) < threshold:
        return False
    extensions = recent_renames['local_title'].str.split('.').str[-1]
    return len(extensions.unique()) == 1

def fuse_evidence(xgb_prob, micropatterns):
    """
    Fuses the XGBoost probability with micropattern evidence using Bayesian odds.

    Args:
        xgb_prob (float): The initial probability from the XGBoost model (0.0 to 1.0).
        micropatterns (dict): A dictionary of boolean flags, e.g.,
                              {'high_velocity': True, 'systematic_rename': False}.

    Returns:
        float: The fused probability (posterior probability).
    """
    # 1. Convert initial probability to prior odds
    prior_odds = xgb_prob / (1 - xgb_prob + EPSILON)

    # 2. Apply evidence by multiplying by likelihood ratios
    posterior_odds = prior_odds
    for pattern_name, is_detected in micropatterns.items():
        if is_detected:
            ratio = LIKELIHOOD_RATIOS.get(pattern_name, 1.0) # Default to 1.0 if not found
            posterior_odds *= ratio
            print(f"  - Evidence applied: '{pattern_name}'. Odds multiplied by {ratio}.")

    # 3. Convert final odds back to probability
    fused_prob = posterior_odds / (1 + posterior_odds)

    return fused_prob


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Testing Micropattern Detectors & Fusion Logic ---")

    # Create a sample DataFrame of events simulating a ransomware attack
    current_time = time.time()
    events = [{'timestamp': current_time - 70, 'event_type': 'MODIFIED', 'local_title': 'a.doc'}]
    for i in range(60):
        events.append({
            'timestamp': current_time - (30 - i*0.5),
            'event_type': 'RENAMED',
            'local_title': f'file_{i}.encrypted'
        })
    events_df = pd.DataFrame(events)

    # --- Run Detectors ---
    is_high_velocity = detect_high_velocity_modification(events_df, window_seconds=60, threshold=50)
    is_systematic_rename = detect_systematic_renaming(events_df, window_seconds=60, threshold=5)
    
    detected_patterns = {
        'high_velocity': is_high_velocity,
        'systematic_rename': is_systematic_rename
    }
    print(f"\nDetected patterns: {detected_patterns}")

    # --- Test Fusion Logic ---
    initial_prob = 0.60 # Assume XGBoost is moderately suspicious
    print(f"\nTesting fusion with initial XGBoost probability = {initial_prob:.2f}")

    final_prob = fuse_evidence(initial_prob, detected_patterns)
    print(f"Final fused probability: {final_prob:.4f} (Expected to be high)")

    # Test a benign case
    initial_prob_benign = 0.05 # Assume XGBoost sees a benign event
    benign_patterns = {'high_velocity': False, 'systematic_rename': False}
    print(f"\nTesting fusion with benign case, initial probability = {initial_prob_benign:.2f}")
    
    final_prob_benign = fuse_evidence(initial_prob_benign, benign_patterns)
    print(f"Final fused probability (benign case): {final_prob_benign:.4f} (Expected to be low)")