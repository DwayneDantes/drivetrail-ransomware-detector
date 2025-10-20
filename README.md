# DriveTrail: A Real-Time Ransomware Detection Agent for Google Drive

## Table of Contents
- [Project Description](#project-description)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

## Project Description

DriveTrail is a proof-of-concept security agent designed to detect ransomware activity within a local Google Drive folder in near real-time. Unlike traditional signature-based antivirus, DriveTrail employs a multi-layered machine learning approach to identify malicious *behavioral patterns* associated with ransomware attacks, such as high-velocity file modification, systematic renaming, and the use of high-entropy filenames.

The system operates by safely monitoring the Google Drive client's metadata database, feeding a stream of file events into a sophisticated inference pipeline that combines a per-event classifier (XGBoost) with a temporal narrative model (Hidden Markov Model) to reconstruct and understand the "story" of an attack as it unfolds.

## System Architecture

The agent uses a multi-layered inference pipeline to achieve high-accuracy, context-aware detection:

1.  **Live Metadata Watcher**: A lightweight sensor that continuously monitors the Google Drive `metadata_sqlite_db` for file changes (Creations, Modifications, Renames, Deletions) in a safe, read-only mode.
2.  **Feature Extractor**: Translates raw file events into a rich 12-dimensional feature vector, including features like filename entropy and extension similarity.
3.  **Per-Event Classifier (XGBoost)**: A gradient-boosted decision tree model that assigns an initial risk score to every individual file event.
4.  **Evidence Fusion Engine (Bayesian)**: Combines the XGBoost score with behavioral "micropatterns" (e.g., a burst of renames) to produce a more reliable, fused probability score.
5.  **Temporal Narrative Model (HMM)**: A Hidden Markov Model that analyzes the sequence of fused scores to understand the evolving narrative, inferring whether the system is in a `Benign` or `Malicious` state.

## Features

- **Behavioral Detection**: Identifies ransomware based on its actions, not just its file hash.
- **Real-Time Monitoring**: Polls for file system changes every few seconds.
- **Explainable AI (XAI)**: Uses SHAP (SHapley Additive exPlanations) to identify the key features driving a model's prediction, ensuring the system is transparent and trustworthy.
- **Narrative Reconstruction**: The HMM is designed to understand the sequence of an attack, enabling the generation of timeline-based alerts.
- **Safe and Non-Intrusive**: Interacts with the Google Drive database in a read-only mode, ensuring no interference with the live sync client.

## Setup and Installation

Follow these steps to set up the project environment and run the DriveTrail agent.

### 1. Prerequisites

- **Python**: Version 3.10 or higher.
- **Git**: For cloning the repository.
- **Google Drive for Desktop**: The agent monitors the live database created by this application. You must have it installed, configured, and running.

    - **To install Google Drive for Desktop:**
        1. Go to the official download page: [https://www.google.com/drive/download/](https://www.google.com/drive/download/)
        2. Download the installer for Windows.
        3. Run the installer and follow the on-screen instructions to sign in with your Google account and set up your local Drive folder.

### 2. Clone the Repository

Open a terminal or Command Prompt and clone this repository to your local machine.

```bash
git clone https://github.com/YourUsername/drivetrail-ransomware-detector.git
cd drivetrail-ransomware-detector
