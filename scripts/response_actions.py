import os
import sys
import subprocess
import psutil
import json
import time
from pathlib import Path
from datetime import datetime
from plyer import notification
import logging

# --- Configuration ---
ALERT_LOG_PATH = os.path.join('data', 'alerts.json')
QUARANTINE_PATH = os.path.join('data', 'quarantine')

# Threat level thresholds
THREAT_LEVELS = {
    'LOW': (0.3, 0.5),      # Suspicious but not critical
    'MEDIUM': (0.5, 0.75),   # Likely threat, warn user
    'HIGH': (0.75, 0.9),     # High confidence, pause sync
    'CRITICAL': (0.9, 1.0)   # Emergency, immediate action
}

# Setup logging
logging.basicConfig(
    filename=os.path.join('data', 'response_actions.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DriveResponseSystem:
    """
    Handles all response actions when ransomware is detected.
    """
    
    def __init__(self):
        self.drive_process_name = "GoogleDriveFS.exe"
        self.sync_paused = False
        self.quarantine_enabled = True
        
        # Create necessary directories
        os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
        os.makedirs(QUARANTINE_PATH, exist_ok=True)
        
    def get_threat_level(self, probability):
        """Determines threat level based on probability score."""
        for level, (low, high) in THREAT_LEVELS.items():
            if low <= probability < high:
                return level
        return 'LOW'
    
    def find_drive_process(self):
        """
        Finds the Google Drive process if it's running.
        Returns the psutil.Process object or None.
        """
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == self.drive_process_name:
                    return proc
            return None
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logging.error(f"Error finding Drive process: {e}")
            return None
    
    def pause_drive_sync(self, method='suspend'):
        """
        Pauses Google Drive sync using various methods.
        
        Args:
            method (str): 'suspend' (pause process), 'quit' (close Drive), 
                         or 'network' (block network access)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.sync_paused:
            logging.info("Drive sync already paused.")
            return True
        
        drive_proc = self.find_drive_process()
        if not drive_proc:
            logging.warning("Google Drive process not found. May already be closed.")
            return False
        
        try:
            if method == 'suspend':
                # Suspend the process (Windows only)
                if sys.platform == 'win32':
                    drive_proc.suspend()
                    self.sync_paused = True
                    logging.info(f"Drive process (PID: {drive_proc.pid}) suspended successfully.")
                    return True
                else:
                    logging.error("Process suspension only works on Windows.")
                    return False
                    
            elif method == 'quit':
                # Gracefully terminate Drive
                drive_proc.terminate()
                drive_proc.wait(timeout=10)
                self.sync_paused = True
                logging.info(f"Drive process (PID: {drive_proc.pid}) terminated successfully.")
                return True
                
            elif method == 'network':
                # Block network access using Windows Firewall
                if sys.platform == 'win32':
                    exe_path = drive_proc.exe()
                    rule_name = "DriveTrail_EmergencyBlock"
                    cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=out program="{exe_path}" action=block'
                    
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.sync_paused = True
                        logging.info("Network access blocked for Drive via Windows Firewall.")
                        return True
                    else:
                        logging.error(f"Failed to block network: {result.stderr}")
                        return False
                else:
                    logging.error("Network blocking only implemented for Windows.")
                    return False
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            logging.error(f"Failed to pause Drive sync: {e}")
            return False
    
    def resume_drive_sync(self):
        """
        Resumes Google Drive sync if it was paused.
        """
        if not self.sync_paused:
            logging.info("Drive sync was not paused, nothing to resume.")
            return True
        
        drive_proc = self.find_drive_process()
        if drive_proc:
            try:
                drive_proc.resume()
                self.sync_paused = False
                logging.info(f"Drive process (PID: {drive_proc.pid}) resumed successfully.")
                return True
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logging.error(f"Failed to resume Drive sync: {e}")
                return False
        else:
            # Process was terminated, need to restart Drive
            logging.info("Drive process not found. User needs to restart Google Drive manually.")
            return False
    
    def remove_firewall_block(self):
        """Removes the emergency firewall block if it was created."""
        if sys.platform == 'win32':
            rule_name = "DriveTrail_EmergencyBlock"
            cmd = f'netsh advfirewall firewall delete rule name="{rule_name}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("Firewall block removed successfully.")
                return True
            else:
                logging.warning(f"Could not remove firewall block: {result.stderr}")
                return False
        return False
    
    def send_alert(self, threat_level, probability, event_details, affected_files_count):
        """
        Sends a Windows toast notification to alert the user.
        
        Args:
            threat_level (str): LOW, MEDIUM, HIGH, or CRITICAL
            probability (float): The fused probability score
            event_details (dict): Details about the triggering event
            affected_files_count (int): Number of files affected in recent window
        """
        # Define notification content based on threat level
        titles = {
            'LOW': '⚠️ DriveTrail: Suspicious Activity',
            'MEDIUM': '🟡 DriveTrail: Potential Threat Detected',
            'HIGH': '🟠 DriveTrail: High Risk - Sync Paused',
            'CRITICAL': '🚨 DriveTrail: RANSOMWARE DETECTED!'
        }
        
        messages = {
            'LOW': f'Monitoring unusual file activity. Confidence: {probability:.1%}',
            'MEDIUM': f'Potential ransomware behavior detected.\n{affected_files_count} files affected. Confidence: {probability:.1%}',
            'HIGH': f'High-risk activity detected!\nDrive sync has been PAUSED.\n{affected_files_count} files may be at risk.\nConfidence: {probability:.1%}',
            'CRITICAL': f'EMERGENCY: Ransomware attack in progress!\nDrive sync BLOCKED.\n{affected_files_count} files affected.\nConfidence: {probability:.1%}\n\nDO NOT close this notification!'
        }
        
        try:
            notification.notify(
                title=titles.get(threat_level, titles['LOW']),
                message=messages.get(threat_level, messages['LOW']),
                app_name='DriveTrail Protection',
                timeout=30 if threat_level in ['LOW', 'MEDIUM'] else 0  # Critical alerts don't auto-dismiss
            )
            logging.info(f"Alert sent: {threat_level} threat detected")
            
            # Also log to alerts file for user review
            self.log_alert(threat_level, probability, event_details, affected_files_count)
            
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")
            # Fallback: print to console
            print(f"\n{'='*60}")
            print(f"  {titles.get(threat_level, 'ALERT')}")
            print(f"  {messages.get(threat_level, 'Threat detected')}")
            print(f"{'='*60}\n")
    
    def log_alert(self, threat_level, probability, event_details, affected_files_count):
        """Logs alert details to a JSON file for user review and forensics."""
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'threat_level': threat_level,
            'probability': probability,
            'affected_files_count': affected_files_count,
            'triggering_event': event_details,
            'sync_paused': self.sync_paused
        }
        
        # Load existing alerts
        alerts = []
        if os.path.exists(ALERT_LOG_PATH):
            try:
                with open(ALERT_LOG_PATH, 'r') as f:
                    alerts = json.load(f)
            except json.JSONDecodeError:
                alerts = []
        
        # Append new alert
        alerts.append(alert_record)
        
        # Save back to file
        with open(ALERT_LOG_PATH, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        logging.info(f"Alert logged to {ALERT_LOG_PATH}")
    
    def quarantine_file(self, file_path):
        """
        Moves a suspicious file to quarantine folder.
        
        Args:
            file_path (str): Path to the file to quarantine
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.quarantine_enabled:
            return False
        
        try:
            source = Path(file_path)
            if not source.exists():
                logging.warning(f"File not found for quarantine: {file_path}")
                return False
            
            # Create a timestamped quarantine subfolder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            quarantine_subfolder = os.path.join(QUARANTINE_PATH, timestamp)
            os.makedirs(quarantine_subfolder, exist_ok=True)
            
            # Move file to quarantine
            destination = Path(quarantine_subfolder) / source.name
            source.rename(destination)
            
            logging.info(f"File quarantined: {file_path} -> {destination}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to quarantine file {file_path}: {e}")
            return False
    
    def take_action(self, threat_level, probability, event_details, affected_files_count):
        """
        Main method to coordinate response based on threat level.
        
        Args:
            threat_level (str): The assessed threat level
            probability (float): Fused probability score
            event_details (dict): Details of the triggering event
            affected_files_count (int): Number of files affected
        """
        logging.info(f"Taking action for {threat_level} threat (prob={probability:.4f})")
        
        # Always send alert
        self.send_alert(threat_level, probability, event_details, affected_files_count)
        
        # Take additional actions based on threat level
        if threat_level == 'MEDIUM':
            # Just alert, let user decide
            pass
            
        elif threat_level == 'HIGH':
            # Pause sync, but keep process alive
            success = self.pause_drive_sync(method='suspend')
            if success:
                print("\n🛑 Drive sync has been PAUSED for your protection.")
                print("   Review the alert and use 'resume_sync.py' to continue if safe.\n")
            
        elif threat_level == 'CRITICAL':
            # Emergency: pause AND consider quarantine
            success = self.pause_drive_sync(method='suspend')
            if success:
                print("\n🚨 EMERGENCY MODE ACTIVATED 🚨")
                print("   Drive sync has been PAUSED.")
                print("   Potentially encrypted files detected.")
                print("   Check Drive web interface for version history.")
                print("   Run 'resume_sync.py' only after verifying safety.\n")
            
            # Optionally quarantine the most recent suspicious file
            if 'full_path' in event_details and event_details['full_path']:
                self.quarantine_file(event_details['full_path'])
    
    def generate_recovery_report(self):
        """
        Generates a user-friendly report with recovery instructions.
        """
        report_path = os.path.join('data', 'recovery_instructions.txt')
        
        report_content = f"""
================================================================
                 DRIVETRAIL RECOVERY GUIDE                     
================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

WARNING: RANSOMWARE ACTIVITY DETECTED - Drive sync has been paused.

IMMEDIATE STEPS:
1. DO NOT close Google Drive or this application yet
2. Disconnect from the internet if attack is ongoing
3. Run a full system antivirus scan

RECOVERY OPTIONS:

Option 1: Restore from Google Drive Version History
----------------------------------------
- Google Drive keeps 30 days of version history for free accounts
- G Suite accounts have longer retention

Steps:
1. Open Google Drive in your web browser
2. Right-click on affected files
3. Select "Manage versions" or "Version history"
4. Restore to the last version BEFORE the attack

Option 2: Check Drive Trash
----------------------------------------
If files were deleted, check:
https://drive.google.com/drive/trash

Option 3: Review Quarantined Files
----------------------------------------
DriveTrail moved suspicious files to:
{os.path.abspath(QUARANTINE_PATH)}

BEFORE RESUMING SYNC:
[ ] Verify ransomware has been removed from your system
[ ] Check that encrypted files are not still present
[ ] Review the alerts log: {os.path.abspath(ALERT_LOG_PATH)}

To resume Drive sync:
Run the command: python scripts/resume_sync.py

For support: Check logs at data/response_actions.log
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"Recovery report generated: {report_path}")
        print(f"\n📄 Recovery instructions saved to: {report_path}")
        
        return report_path


# --- Example Usage & Testing ---
if __name__ == '__main__':
    print("--- Testing DriveTrail Response System ---\n")
    
    # Initialize the response system
    response = DriveResponseSystem()
    
    # Simulate different threat scenarios
    test_event = {
        'stable_id': 12345,
        'event_type': 'RENAMED',
        'local_title': 'important_document.encrypted',
        'full_path': 'My Drive/Documents/important_document.encrypted'
    }
    
    # Test 1: LOW threat (just monitoring)
    print("Test 1: LOW threat scenario")
    response.take_action('LOW', 0.35, test_event, 3)
    time.sleep(2)
    
    # Test 2: MEDIUM threat (alert user)
    print("\nTest 2: MEDIUM threat scenario")
    response.take_action('MEDIUM', 0.65, test_event, 15)
    time.sleep(2)
    
    # Test 3: HIGH threat (pause sync)
    print("\nTest 3: HIGH threat scenario")
    response.take_action('HIGH', 0.82, test_event, 47)
    time.sleep(2)
    
    # Generate recovery report
    print("\nGenerating recovery report...")
    response.generate_recovery_report()
    
    print("\n--- Test complete. Check 'data/alerts.json' and notifications ---")
    print("Note: Drive sync may have been paused. Run 'resume_sync.py' if needed.")