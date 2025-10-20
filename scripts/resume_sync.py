"""
resume_sync.py - Safely resume Google Drive sync after a threat alert

This script should only be run after:
1. Verifying ransomware has been removed
2. Checking that no encrypted files remain
3. Ensuring your system is clean (run antivirus scan)
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from scripts.response_actions import DriveResponseSystem
import json

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          DriveTrail - Resume Sync Safety Check              ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    response = DriveResponseSystem()
    
    # Check if sync is actually paused
    if not response.sync_paused:
        print("ℹ️  Drive sync is not currently paused by DriveTrail.")
        print("   No action needed.\n")
        return
    
    # Load recent alerts to show user what happened
    alerts_path = os.path.join('data', 'alerts.json')
    if os.path.exists(alerts_path):
        with open(alerts_path, 'r') as f:
            alerts = json.load(f)
        
        if alerts:
            latest_alert = alerts[-1]
            print(f"⚠️  Latest Alert:")
            print(f"   Time: {latest_alert['timestamp']}")
            print(f"   Threat Level: {latest_alert['threat_level']}")
            print(f"   Confidence: {latest_alert['probability']:.1%}")
            print(f"   Files Affected: {latest_alert['affected_files_count']}\n")
    
    # Safety checklist
    print("Before resuming sync, please confirm:\n")
    print("□ Have you run a full antivirus/antimalware scan?")
    print("□ Have you verified no encrypted files remain on your system?")
    print("□ Have you checked Google Drive web interface for damage?")
    print("□ Have you reviewed the recovery instructions?")
    print("□ Are you confident the threat has been eliminated?\n")
    
    # User confirmation
    confirm = input("Type 'RESUME' (all caps) to continue, or anything else to cancel: ")
    
    if confirm.strip() == 'RESUME':
        print("\nAttempting to resume Drive sync...")
        
        # Try to resume
        success = response.resume_drive_sync()
        
        if success:
            print("✅ Drive sync has been resumed successfully.")
            print("   Monitor DriveTrail agent for any further suspicious activity.\n")
        else:
            print("⚠️  Could not resume automatically.")
            print("   You may need to restart Google Drive manually.\n")
        
        # Clean up firewall rules if they were created
        response.remove_firewall_block()
        
    else:
        print("\n❌ Sync resume cancelled.")
        print("   Drive will remain paused for your safety.")
        print("   Review logs and recovery instructions before trying again.\n")

if __name__ == "__main__":
    main()