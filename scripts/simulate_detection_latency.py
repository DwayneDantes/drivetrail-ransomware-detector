import numpy as np
import pandas as pd
import json
import os

RESULTS_DIR = 'paper_results'

def simulate_ransomware_scenarios():
    """
    Simulate detection latency for different ransomware behaviors.
    Based on typical ransomware encryption speeds from literature.
    """
    
    scenarios = {
        'Fast Burst Encryption': {
            'description': 'Aggressive ransomware (e.g., LockBit-style)',
            'files_per_second': 50,  # Very fast
            'total_files': 500,
            'detection_threshold_files': 25,  # Detected after 25 files
            'polling_interval': 5  # DriveTrail polls every 5 seconds
        },
        'Moderate Encryption': {
            'description': 'Standard ransomware (e.g., Conti-style)',
            'files_per_second': 20,
            'total_files': 300,
            'detection_threshold_files': 20,
            'polling_interval': 5
        },
        'Stealthy Encryption': {
            'description': 'Slow, evasive ransomware',
            'files_per_second': 5,
            'total_files': 200,
            'detection_threshold_files': 30,
            'polling_interval': 5
        },
        'Systematic Deletion': {
            'description': 'Wiper-style ransomware',
            'files_per_second': 30,
            'total_files': 400,
            'detection_threshold_files': 20,
            'polling_interval': 5
        }
    }
    
    results = []
    
    for scenario_name, params in scenarios.items():
        # Calculate detection latency
        time_to_threshold = params['detection_threshold_files'] / params['files_per_second']
        
        # Add polling delay (worst case: just missed last poll)
        detection_latency = time_to_threshold + params['polling_interval']
        
        # Calculate mean file loss
        files_at_detection = params['files_per_second'] * detection_latency
        mean_file_loss_pct = (files_at_detection / params['total_files']) * 100
        
        # Clip to 100%
        mean_file_loss_pct = min(mean_file_loss_pct, 100.0)
        
        result = {
            'scenario': scenario_name,
            'description': params['description'],
            'detection_latency_seconds': round(detection_latency, 2),
            'files_encrypted_before_alert': int(files_at_detection),
            'total_files': params['total_files'],
            'mean_file_loss_percent': round(mean_file_loss_pct, 2),
            'alert_effectiveness': 'Excellent' if mean_file_loss_pct < 10 else 
                                  ('Good' if mean_file_loss_pct < 30 else 'Moderate')
        }
        
        results.append(result)
    
    return results

def main():
    print("="*70)
    print("  DETECTION LATENCY SIMULATION")
    print("="*70)
    
    print("\nℹ️  Simulating detection latency based on typical ransomware speeds...")
    
    results = simulate_ransomware_scenarios()
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, 'detection_latency_simulation.json')
    
    with open(output_file, 'w') as f:
        json.dump({'scenarios': results}, f, indent=2)
    
    print(f"💾 Results saved: {output_file}\n")
    
    # Print table
    print("="*70)
    print("  DETECTION LATENCY RESULTS")
    print("="*70)
    print(f"{'Scenario':<30} {'Latency (s)':<15} {'MFL (%)':<12} {'Effectiveness'}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['scenario']:<30} "
              f"{r['detection_latency_seconds']:<15.2f} "
              f"{r['mean_file_loss_percent']:<12.2f} "
              f"{r['alert_effectiveness']}")
    
    print("="*70)
    
    # Calculate average
    avg_latency = np.mean([r['detection_latency_seconds'] for r in results])
    avg_mfl = np.mean([r['mean_file_loss_percent'] for r in results])
    
    print(f"\n📊 Average Detection Latency: {avg_latency:.2f} seconds")
    print(f"📊 Average Mean File Loss: {avg_mfl:.2f}%")
    print("\n✅ Simulation complete!")

if __name__ == "__main__":
    main()