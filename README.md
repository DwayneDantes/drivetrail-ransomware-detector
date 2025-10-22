# DriveTrail - Ransomware Detection for Google Drive 🛡️

**Real-time behavioral ransomware detection using multi-layered machine learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-proof--of--concept-yellow.svg)]()

---

## 📋 Table of Contents

1. [Project Description](#project-description)
2. [System Architecture](#system-architecture)
3. [Key Features](#key-features)
4. [Prerequisites](#prerequisites)
5. [Installation Guide](#installation-guide)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [Future Work](#future-work)
13. [License](#license)

---

## 🎯 Project Description

**DriveTrail** is a proof-of-concept security agent that detects ransomware activity within your local Google Drive folder in **near real-time**. Unlike traditional signature-based antivirus solutions, DriveTrail employs a sophisticated **multi-layered machine learning approach** to identify malicious behavioral patterns characteristic of ransomware attacks:

- 🚀 **High-velocity file modification** (rapid encryption of many files)
- 🔄 **Systematic file renaming** (mass extension changes)
- 🔐 **High-entropy filenames** (encrypted/randomized names)
- 📊 **Suspicious directory traversal patterns**

### Why DriveTrail?

**Traditional Antivirus:**
- ❌ Relies on known malware signatures
- ❌ Misses zero-day ransomware variants
- ❌ Can't stop cloud propagation
- ❌ No behavioral analysis

**DriveTrail:**
- ✅ Detects based on **behavior**, not signatures
- ✅ Catches **novel ransomware** variants
- ✅ **Pauses Google Drive sync** to prevent cloud propagation
- ✅ Provides **forensic timeline** of attacks
- ✅ **Protects all your devices** by stopping sync before damage spreads

### How It Works

DriveTrail operates by safely monitoring the Google Drive client's metadata database, feeding a stream of file events into a sophisticated inference pipeline that combines:

1. **Per-event classification** (XGBoost)
2. **Behavioral pattern detection** (Bayesian Evidence Fusion)
3. **Temporal sequence analysis** (Hidden Markov Model)

This multi-layered approach reconstructs and understands the "story" of an attack as it unfolds, enabling rapid detection and automated response.

---

## 🏗️ System Architecture

DriveTrail uses a **5-stage inference pipeline** for high-accuracy, context-aware detection:
```
┌─────────────────────────────────────────────────────────────┐
│  1. LIVE METADATA WATCHER                                   │
│  Monitors Google Drive metadata database                    │
│  Detects: CREATED, MODIFIED, RENAMED, DELETED events        │
└─────────────────┬───────────────────────────────────────────┘
                  │ File Events Stream
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  2. FEATURE EXTRACTOR                                       │
│  Converts events → 12-dimensional feature vectors           │
│  Features: entropy, path depth, extension similarity, etc.  │
└─────────────────┬───────────────────────────────────────────┘
                  │ Feature Vectors
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  3. XGBoost CLASSIFIER                                      │
│  Per-event risk scoring                                     │
│  Output: Initial probability (0.0 - 1.0)                    │
└─────────────────┬───────────────────────────────────────────┘
                  │ Risk Scores
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  4. BAYESIAN EVIDENCE FUSION                                │
│  Combines XGBoost + Micropattern Detection                  │
│  Patterns: High velocity, systematic renaming               │
│  Output: Fused probability                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │ Fused Scores
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  5. HIDDEN MARKOV MODEL (HMM)                               │
│  Temporal sequence analysis                                 │
│  Infers system state: Benign vs Malicious                   │
│  Output: State sequence + Alert decision                    │
└─────────────────┬───────────────────────────────────────────┘
                  │ Final Decision
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  RESPONSE SYSTEM                                            │
│  • Windows notifications                                    │
│  • Pause Google Drive sync                                  │
│  • Quarantine suspicious files                              │
│  • Generate recovery instructions                           │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### Detection Capabilities

- 🎯 **Behavioral Detection**: Identifies ransomware based on actions, not signatures
- ⚡ **Real-Time Monitoring**: Polls for file system changes every 3-5 seconds
- 🧠 **Multi-Layer AI**: XGBoost + Bayesian Fusion + HMM for robust detection
- 📊 **Explainable AI (XAI)**: SHAP analysis shows which features drive predictions
- 🔍 **Pattern Recognition**: Detects high-velocity changes, systematic renaming, suspicious paths

### Response & Protection

- 🛑 **Automatic Sync Pause**: Stops Google Drive sync when threats detected
- 🔔 **Windows Notifications**: Immediate visual alerts at different threat levels
- 📁 **File Quarantine**: Isolates suspicious files automatically
- 📋 **Recovery Guidance**: Auto-generates step-by-step recovery instructions
- 🌐 **Cloud Protection**: Prevents encrypted files from syncing to cloud and other devices

### Analysis & Forensics

- 📈 **Narrative Reconstruction**: Understands the sequence and timeline of attacks
- 🕐 **Forensic Timeline**: Complete second-by-second attack reconstruction
- 📊 **Visual Analytics**: Charts showing encryption rates, affected folders, file types
- 💾 **Audit Trail**: Complete logging of all events and decisions
- 🔬 **Attack Pattern Analysis**: Identifies ransomware behavior characteristics

### Safety & Reliability

- 🔒 **Safe Operation**: Read-only interaction with Google Drive database
- 🚫 **Non-Intrusive**: No interference with live sync client
- 🎯 **Low False Positives**: Optimized thresholds via Bayesian optimization
- 🔄 **Adaptive Learning**: Can re-optimize configurations based on your usage patterns

---

## 📋 Prerequisites

### Required Software

#### 1. **Python 3.10 or Higher**

**Check if you have Python:**
```bash
python --version
# Should show: Python 3.10.x or higher
```

**If not installed:**
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
  - ✅ **IMPORTANT**: Check "Add Python to PATH" during installation
- **macOS**: `brew install python@3.10`
- **Linux**: `sudo apt install python3.10 python3.10-venv`

#### 2. **Git** (for cloning repository)

**Check if you have Git:**
```bash
git --version
```

**If not installed:**
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **macOS**: `brew install git` or install Xcode Command Line Tools
- **Linux**: `sudo apt install git`

#### 3. **Google Drive for Desktop**

**This is CRITICAL** - DriveTrail monitors the metadata database created by Google Drive.

**Installation:**
1. Go to [google.com/drive/download](https://www.google.com/drive/download/)
2. Download "Drive for Desktop"
3. Run the installer
4. Sign in with your Google account
5. Choose folders to sync
6. Wait for initial sync to complete

**Verify Installation:**
- Check system tray for Google Drive icon
- Ensure you have a Google Drive folder on your computer
- Common locations:
  - Windows: `C:\Users\YourName\Google Drive`
  - macOS: `/Users/YourName/Google Drive`
  - Linux: `/home/yourname/GoogleDrive`

#### 4. **Administrator Privileges** (Windows)

For full functionality (pausing Drive sync), you'll need to run scripts with administrator rights.

### System Requirements

- **OS**: Windows 10/11 (primary), macOS 10.15+, Linux (Ubuntu 20.04+)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space (for models and logs)
- **Internet**: Required for Drive sync and downloading dependencies

---

## 🚀 Installation Guide

### Step 1: Clone the Repository

Open **Command Prompt** (Windows) or **Terminal** (macOS/Linux):
```bash
# Clone the repository
git clone https://github.com/YourUsername/drivetrail.git

# Navigate into the project directory
cd drivetrail
```

### Step 2: Create a Virtual Environment

**Why use a virtual environment?**
- Isolates project dependencies
- Prevents conflicts with other Python projects
- Makes the project portable

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) in your prompt
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your prompt
```

**Verify activation:**
```bash
# Should show path inside venv folder
which python  # macOS/Linux
where python  # Windows
```

### Step 3: Install Dependencies

With the virtual environment activated:
```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# This will install:
# - pandas, numpy (data processing)
# - xgboost, scikit-learn (machine learning)
# - hmmlearn (Hidden Markov Models)
# - psutil, plyer (system monitoring & notifications)
# - optuna, plotly (hyperparameter optimization)
# - And many more...
```

**Installation time:** 2-5 minutes depending on your internet speed.

**Verify installation:**
```bash
# Check if key packages installed
python -c "import xgboost; import optuna; print('✅ All packages installed!')"
```

### Step 4: Prepare Training Data

DriveTrail needs labeled data to train its models.

**Option A: Use the Provided Dataset**

If you have the `Ransomware_Data.csv` file:
```bash
# Create data directories
mkdir data\raw        # Windows
mkdir -p data/raw     # macOS/Linux

# Place Ransomware_Data.csv in data/raw/
# Should be at: data/raw/Ransomware_Data.csv
```

**Option B: Use Your Own Dataset**

Your CSV should have these columns:
- `Ware Type` - Label ('good' or 'ransom')
- `File_Delete_archived` - Binary feature
- `File_created` - Binary feature
- `process-related` - Binary feature
- `network-related` - Binary feature
- `file-related` - Binary feature
- `suspicious_path` - Binary feature
- `system_executable` - Binary feature
- `path_length` - Numeric feature
- `directory_depth` - Numeric feature
- `process_name_length` - Numeric feature
- `extension_similarity` - Numeric feature (0.0-1.0)
- `file_name_entropy` - Numeric feature

### Step 5: Process Training Data
```bash
# Convert raw data to training-ready format
python scripts/csu_to_drive_synthesizer.py
```

**Expected output:**
```
--- Starting Data Synthesis Script ---
Loading raw data from 'data/raw/Ransomware_Data.csv'...
Loaded 10000 rows and 13 columns.
Selecting the 12 relevant columns for Phase 1...
Encoding 'Ware Type' label (good=0, ransom=1)...
Processed data has 10000 rows and 13 columns.
Final columns: ['label', 'File_Delete_archived', ...]
Saving processed data to 'data/training_features.parquet'...
--- Script finished successfully! ---
```

**Result:** `data/training_features.parquet` created

### Step 6: Train Models

**Train XGBoost Classifier:**
```bash
python scripts/train_and_evaluate.py
```

**Expected output:**
```
--- Starting Full Training & Evaluation Script ---
Loading data from 'data/training_features.parquet'...
Splitting data into training and testing sets...
Training the XGBoost classifier...
Model training complete.

--- Model Performance Report ---
Classification Report:
              precision    recall  f1-score   support
        good       0.96      0.98      0.97      1000
      ransom       0.98      0.96      0.97      1000

[Confusion Matrix displayed]
[SHAP visualizations displayed]

Saving the trained model to 'models/xgb_drivetrail.model'...
--- Script finished successfully! ---
```

**Train Hidden Markov Model:**
```bash
python scripts/train_hmm.py
```

**Expected output:**
```
--- Starting HMM Model Training Script (hmmlearn) ---
Loading data from 'data/training_features.parquet'...
Loaded 5000 benign and 5000 malicious samples.
Generating 500 sequences of length 15 for each class...
Defining and training GaussianHMM with 2 states...
Model training complete.

Discovered States (based on mean file_name_entropy):
  - Benign State (ID 0): Mean = 2.1234
  - Malicious State (ID 1): Mean = 4.5678

Saving the trained model to 'models/hmm_drivetrail.joblib'...
--- HMM Script finished successfully! ---
```

**Training time:** 5-10 minutes total

**Verify models created:**
```bash
ls models/
# Should see:
# xgb_drivetrail.model
# hmm_drivetrail.joblib
```

### Step 7: (Optional) Optimize Configuration

For best performance, run hyperparameter optimization:
```bash
# Install Optuna if not already installed
pip install optuna plotly kaleido

# Run optimization (takes 15-30 minutes)
python scripts/optimize_config_optuna.py
# Enter: 100 (when prompted for number of trials)

# Apply learned configuration
python scripts/apply_config.py
# Type: yes (when prompted)

# Retrain HMM with optimized parameters
python scripts/train_hmm.py
```

**This step is optional but highly recommended for production use.**

---

## ⚙️ Configuration

### Default Configuration Files

| File | Purpose |
|------|---------|
| `scripts/bayes_fusion.py` | Likelihood ratios and micropattern thresholds |
| `scripts/run_agent.py` | Alert thresholds and polling intervals |
| `scripts/train_hmm.py` | HMM parameters |
| `data/learned_config.json` | Learned optimal configuration (if optimized) |

### Key Configuration Parameters

**Alert Thresholds** (`scripts/run_agent.py`):
```python
ALERT_THRESHOLD = 0.50      # Minimum probability to trigger alert
PAUSE_THRESHOLD = 0.75      # Minimum probability to pause sync
CRITICAL_THRESHOLD = 0.90   # Emergency threshold
```

**Micropattern Detection** (`scripts/bayes_fusion.py`):
```python
# High velocity detection
threshold=50                # Number of events in 60 seconds

# Systematic renaming detection
threshold=5                 # Number of renamed files with same extension
```

**For detailed configuration options, see:** [DYNAMIC_CONFIG_GUIDE.md](DYNAMIC_CONFIG_GUIDE.md)

---

## 📖 Usage

### Starting DriveTrail

You need **TWO terminal windows** running simultaneously:

**Terminal 1: Start the Metadata Watcher**
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Start the watcher
python scripts/drive_meta_watcher.py
```

**Expected output:**
```
--- Starting Drive Meta Watcher (v2 - with Path Reconstruction) ---
Event database initialized at 'data\events.db'
Monitoring database: C:\Users\...\AppData\Local\Google\DriveFS\...\metadata_sqlite_db
Logging events to: data\events.db
Polling every 5 seconds. Press Ctrl+C to stop.
Establishing initial baseline snapshot...
```

**Terminal 2: Start the Detection Agent**
```bash
# Activate virtual environment (in new terminal)
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Start the agent (as Administrator on Windows for full functionality)
python scripts/run_agent.py
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════╗
║          DriveTrail Detection Agent - ACTIVE                ║
╚══════════════════════════════════════════════════════════════╝

Loading models...
  - XGBoost model loaded.
  - HMM model loaded.
HMM malicious state identified as State 1 (Mean Entropy: 4.23)
Agent started. Polling every 3 seconds.
Alert threshold: 50.0% | Pause threshold: 75.0%
Press Ctrl+C to stop.
```

**The system is now monitoring your Drive!** 🎉

### Testing the System

**Run a safe ransomware simulation:**
```bash
# In a third terminal
python scripts/test_ransomware_simulation.py
```

Follow the menu:
1. Choose option **2** (Fast Ransomware Encryption)
2. Watch Terminal 2 for detection alerts
3. You should see a Windows notification
4. Drive sync should be paused automatically

### Resuming After Alert

If Drive sync was paused:
```bash
python scripts/resume_sync.py
# Follow the safety checklist
# Type: RESUME (when safe)
```

### Viewing Activity & Forensics

**Quick timeline view:**
```bash
python scripts/quick_timeline.py
```

**Full forensic analysis:**
```bash
python scripts/forensic_timeline.py
```

Generates:
- `data/forensics/attack_1_timeline.txt` - Detailed text report
- `data/forensics/attack_1_visualization.png` - Visual charts
- `data/forensics/attack_1_events.csv` - Raw data export

---

## 📁 Project Structure
```
drivetrail/
├── data/
│   ├── raw/
│   │   └── Ransomware_Data.csv          # Training dataset
│   ├── events.db                         # SQLite event log
│   ├── alerts.json                       # Alert history
│   ├── learned_config.json               # Optimized configuration
│   ├── recovery_instructions.txt         # Auto-generated recovery guide
│   ├── quarantine/                       # Quarantined suspicious files
│   └── forensics/                        # Forensic analysis reports
├── models/
│   ├── xgb_drivetrail.model             # Trained XGBoost model
│   └── hmm_drivetrail.joblib            # Trained HMM model
├── scripts/
│   ├── drive_meta_watcher.py            # ① Metadata monitoring
│   ├── feature_extractor.py             # ② Feature engineering
│   ├── bayes_fusion.py                  # ④ Evidence fusion
│   ├── run_agent.py                     # Main detection agent
│   ├── response_actions.py              # Response system
│   ├── resume_sync.py                   # Resume utility
│   ├── train_and_evaluate.py            # XGBoost training
│   ├── train_hmm.py                     # HMM training
│   ├── optimize_config_optuna.py        # Hyperparameter optimization
│   ├── apply_config.py                  # Config application
│   ├── test_ransomware_simulation.py    # Safe testing tool
│   ├── forensic_timeline.py             # Forensic analysis
│   └── quick_timeline.py                # Quick event viewer
├── venv/                                 # Virtual environment (created by you)
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── LICENSE                               # Project license
```

---

## 🚀 Advanced Features

### Hyperparameter Optimization with Optuna
```bash
python scripts/optimize_config_optuna.py
```

Uses Bayesian optimization to find optimal configurations. See [OPTUNA_OPTIMIZATION_GUIDE.md](OPTUNA_OPTIMIZATION_GUIDE.md) for details.

### Forensic Timeline Reconstruction
```bash
python scripts/forensic_timeline.py
```

Reconstructs complete attack timelines with visualizations. See [FORENSICS_GUIDE.md](FORENSICS_GUIDE.md) for details.

### Dynamic Configuration Learning
```bash
python scripts/optimize_config.py
python scripts/apply_config.py
```

Learns optimal settings from your data. See [DYNAMIC_CONFIG_GUIDE.md](DYNAMIC_CONFIG_GUIDE.md) for details.

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'X'"

**Solution:**
```bash
# Make sure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstall requirements
pip install -r requirements.txt
```

### "Google Drive database could not be found"

**Solution:**
- Ensure Google Drive for Desktop is installed and running
- Check if Drive icon is in system tray
- Verify you have a local Drive folder
- Try restarting Google Drive application

### "Permission denied" when pausing Drive

**Solution:**
- Run Command Prompt as Administrator (Windows)
- Right-click → "Run as Administrator"
- Then run: `python scripts/run_agent.py`

### Notifications not appearing

**Solution:**
- Windows: Settings → System → Notifications → Enable for Python
- Test: `python scripts/response_actions.py`

### High false positive rate

**Solution:**
```bash
# Run optimization to tune for your environment
python scripts/optimize_config_optuna.py
python scripts/apply_config.py
```

For more troubleshooting, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md).

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🔮 Future Work

### Planned Features

- [ ] **Multi-platform support** - Full support for macOS and Linux
- [ ] **Web dashboard** - Real-time monitoring interface
- [ ] **Email/SMS alerts** - Remote notifications
- [ ] **Automatic file restoration** - One-click recovery from Drive history
- [ ] **Multi-device coordination** - Shared threat intelligence across devices
- [ ] **Cloud threat intelligence** - Learn from community detections
- [ ] **Advanced ransomware families** - Detection of specific variants (WannaCry, Ryuk, etc.)
- [ ] **Integration with EDR systems** - Export to SIEM/EDR platforms

### Research Directions

- Deep learning models (LSTM, Transformers) for sequence analysis
- Federated learning for privacy-preserving model updates
- Behavioral biometrics for user verification
- Integration with Windows Event Logs
- Network traffic analysis correlation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- CSE-CIC-IDS2018 dataset for training data
- Optuna team for hyperparameter optimization framework
- XGBoost and scikit-learn communities
- Google Drive for Desktop for making metadata accessible

---

## 📞 Support

- **Documentation**: Check the `docs/` folder for detailed guides
- **Issues**: Report bugs via [GitHub Issues](https://github.com/YourUsername/drivetrail/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/YourUsername/drivetrail/discussions)

---

## ⚠️ Disclaimer

DriveTrail is a **proof-of-concept research project**. While functional, it should not be considered a replacement for:
- Professional antivirus software
- Regular system backups
- Security best practices
- Enterprise-grade security solutions

Always maintain:
- ✅ Updated antivirus software
- ✅ Regular offline backups
- ✅ Operating system patches
- ✅ Security awareness training

---

**Built with ❤️ for safer cloud storage**

**🛡️ Protect Your Drive. Detect the Threat. Stop the Spread.**
