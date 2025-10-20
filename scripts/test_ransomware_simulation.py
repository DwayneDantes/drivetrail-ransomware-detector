"""
test_ransomware_simulation.py - Safe ransomware behavior simulator

⚠️ WARNING: This script simulates ransomware behavior for TESTING ONLY
Only run this in a dedicated TEST folder within your Google Drive!

This helps you verify that DriveTrail's detection and response systems work correctly.
"""

import os
import time
import random
import string
from pathlib import Path

# --- Configuration ---
# IMPORTANT: Change this to a safe test location!
TEST_FOLDER_NAME = "DriveTrail_TEST_DO_NOT_DELETE"

def get_drive_root():
    """Attempts to find the Google Drive root folder."""
    # Common locations for Google Drive
    possible_paths = [
        os.path.join(os.environ.get('USERPROFILE', ''), 'Google Drive'),
        os.path.join(os.environ.get('HOME', ''), 'Google Drive'),
        os.path.join(os.environ.get('USERPROFILE', ''), 'GoogleDrive'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def create_test_folder():
    """Creates a safe test folder in Google Drive."""
    drive_root = get_drive_root()
    
    if not drive_root:
        print("❌ Could not find Google Drive folder.")
        print("   Please enter the full path to your Google Drive folder:")
        drive_root = input("   Path: ").strip()
        
        if not os.path.exists(drive_root):
            print("❌ Invalid path. Exiting.")
            return None
    
    test_folder = os.path.join(drive_root, TEST_FOLDER_NAME)
    os.makedirs(test_folder, exist_ok=True)
    
    print(f"✅ Test folder created: {test_folder}")
    return test_folder

def random_string(length=10):
    """Generates a random string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def simulate_normal_activity(test_folder, num_files=5):
    """Simulates normal file operations (slow, varied)."""
    print(f"\n📝 Simulating NORMAL activity ({num_files} files)...")
    print("   (This should NOT trigger alerts)\n")
    
    for i in range(num_files):
        # Create file
        filename = f"normal_file_{i}.txt"
        filepath = os.path.join(test_folder, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Normal content {random_string()}")
        
        print(f"   ✓ Created: {filename}")
        time.sleep(random.uniform(2, 5))  # Slow, human-like timing
        
        # Maybe modify it
        if random.random() > 0.5:
            with open(filepath, 'a') as f:
                f.write("\nEdited content")
            print(f"   ✓ Modified: {filename}")
            time.sleep(random.uniform(1, 3))
    
    print("✅ Normal activity simulation complete.\n")

def simulate_ransomware_encryption(test_folder, num_files=50, speed='fast'):
    """
    Simulates ransomware encryption behavior.
    
    Args:
        test_folder: Path to test folder
        num_files: Number of files to encrypt
        speed: 'fast' or 'slow' - controls timing
    """
    print(f"\n🚨 Simulating RANSOMWARE behavior ({num_files} files, {speed} speed)...")
    print("   (This SHOULD trigger HIGH/CRITICAL alerts)\n")
    
    # Create original files first
    print("   Creating test files...")
    original_files = []
    for i in range(num_files):
        filename = f"document_{i}.txt"
        filepath = os.path.join(test_folder, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Important document content {random_string(50)}")
        
        original_files.append(filepath)
    
    print(f"   ✓ Created {num_files} test files\n")
    time.sleep(2)
    
    # Now "encrypt" them rapidly
    print("   🔒 Starting 'encryption' process...")
    
    delay = 0.1 if speed == 'fast' else 0.5
    
    for i, original_path in enumerate(original_files):
        # Simulate encryption: rename to .encrypted
        encrypted_path = original_path + '.encrypted'
        
        try:
            os.rename(original_path, encrypted_path)
            print(f"   🔒 [{i+1}/{num_files}] Encrypted: {os.path.basename(encrypted_path)}")
        except Exception as e:
            print(f"   ❌ Error encrypting file: {e}")
        
        time.sleep(delay)
    
    # Create a ransom note (common ransomware behavior)
    ransom_note_path = os.path.join(test_folder, "README_TO_DECRYPT.txt")
    with open(ransom_note_path, 'w') as f:
        f.write("""
========================================
YOUR FILES HAVE BEEN ENCRYPTED! (TEST)
========================================

This is a TEST simulation only.
No actual encryption has occurred.

In a real attack, you would see:
- Instructions to pay ransom
- Bitcoin wallet address
- Threats and deadlines

For testing DriveTrail detection only.
========================================
""")
    print(f"\n   📄 Created ransom note: README_TO_DECRYPT.txt")
    
    print("\n🚨 Ransomware simulation complete!")
    print("   ⚠️  DriveTrail should have detected this and paused Drive sync.\n")

def simulate_systematic_deletion(test_folder, num_files=30):
    """Simulates ransomware deleting files systematically."""
    print(f"\n🗑️  Simulating SYSTEMATIC DELETION ({num_files} files)...")
    print("   (This should trigger alerts)\n")
    
    # Create files
    files_to_delete = []
    for i in range(num_files):
        filename = f"backup_{i}.txt"
        filepath = os.path.join(test_folder, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Backup data {random_string(30)}")
        
        files_to_delete.append(filepath)
    
    time.sleep(1)
    
    # Delete them rapidly
    print("   🗑️  Deleting files...")
    for i, filepath in enumerate(files_to_delete):
        try:
            os.remove(filepath)
            print(f"   🗑️  [{i+1}/{num_files}] Deleted: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(0.1)
    
    print("\n✅ Deletion simulation complete.\n")

def cleanup_test_folder(test_folder):
    """Removes all test files (cleanup after testing)."""
    print(f"\n🧹 Cleaning up test folder: {test_folder}")
    
    confirm = input("   Delete all test files? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        try:
            import shutil
            shutil.rmtree(test_folder)
            print("✅ Test folder deleted successfully.\n")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}\n")
    else:
        print("   Cleanup cancelled. Test files remain.\n")

def main():
    """Main testing interface."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      DriveTrail Ransomware Behavior Simulator (SAFE)        ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    print("⚠️  WARNING: This script will create and modify files in your Google Drive.")
    print("   Make sure the DriveTrail agent is running to see detection in action!\n")
    
    # Create test folder
    test_folder = create_test_folder()
    if not test_folder:
        return
    
    # Main menu
    while True:
        print("\n" + "─"*60)
        print("Select a test scenario:")
        print("─"*60)
        print("1. Normal Activity (should NOT trigger alerts)")
        print("2. Ransomware Encryption - Fast (SHOULD trigger HIGH/CRITICAL)")
        print("3. Ransomware Encryption - Slow (SHOULD trigger alerts)")
        print("4. Systematic Deletion (SHOULD trigger alerts)")
        print("5. Full Test Suite (runs all scenarios)")
        print("6. Cleanup Test Folder")
        print("0. Exit")
        print("─"*60)
        
        choice = input("Enter choice (0-6): ").strip()
        
        if choice == '1':
            simulate_normal_activity(test_folder, num_files=5)
            
        elif choice == '2':
            print("\n⚠️  Starting FAST ransomware simulation in 3 seconds...")
            print("   Watch for DriveTrail alerts!")
            time.sleep(3)
            simulate_ransomware_encryption(test_folder, num_files=50, speed='fast')
            
        elif choice == '3':
            print("\n⚠️  Starting SLOW ransomware simulation in 3 seconds...")
            print("   Watch for DriveTrail alerts!")
            time.sleep(3)
            simulate_ransomware_encryption(test_folder, num_files=30, speed='slow')
            
        elif choice == '4':
            print("\n⚠️  Starting deletion simulation in 3 seconds...")
            print("   Watch for DriveTrail alerts!")
            time.sleep(3)
            simulate_systematic_deletion(test_folder, num_files=30)
            
        elif choice == '5':
            print("\n⚠️  Running FULL TEST SUITE...")
            print("   This will take several minutes.\n")
            
            input("Press Enter to start normal activity test...")
            simulate_normal_activity(test_folder, num_files=5)
            
            input("\nPress Enter to start ransomware encryption test...")
            time.sleep(3)
            simulate_ransomware_encryption(test_folder, num_files=50, speed='fast')
            
            input("\nPress Enter to start deletion test...")
            time.sleep(3)
            simulate_systematic_deletion(test_folder, num_files=30)
            
            print("\n✅ Full test suite complete!")
            print("   Check DriveTrail logs and alerts.\n")
            
        elif choice == '6':
            cleanup_test_folder(test_folder)
            
        elif choice == '0':
            print("\n👋 Exiting simulator. Test files remain in Drive.")
            print("   Remember to clean up the test folder!\n")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                     Testing Complete                         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user.")
        print("   Test files may remain in Drive. Remember to clean up!\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()