"""
optimize_config_optuna.py - Advanced hyperparameter optimization using Optuna

Uses Bayesian optimization to find optimal configurations that:
- Maximize threat detection (recall)
- Minimize false positives (precision)
- Balance detection speed vs accuracy
"""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- Paths ---
TRAINING_DATA_PATH = os.path.join('data', 'training_features.parquet')
EVENT_DB_PATH = os.path.join('data', 'events.db')
CONFIG_OUTPUT_PATH = os.path.join('data', 'learned_config.json')
OPTUNA_DB_PATH = os.path.join('data', 'optuna_study.db')
ANALYSIS_OUTPUT_DIR = os.path.join('data', 'config_analysis')

class OptunaConfigOptimizer:
    """
    Advanced configuration optimizer using Optuna's Bayesian optimization.
    """
    
    def __init__(self):
        self.training_df = None
        self.user_events_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.optimal_config = {}
        
        os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
        
        # User's baseline (learned from their Drive activity)
        self.user_baseline = {
            'event_rate': None,  # events per second
            'typical_burst_size': None,
            'common_extensions': []
        }
    
    def load_data(self):
        """Load training data and user activity."""
        print("📂 Loading data...")
        
        # Load training data
        if not os.path.exists(TRAINING_DATA_PATH):
            print("❌ Training data not found!")
            return False
        
        self.training_df = pd.read_parquet(TRAINING_DATA_PATH)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X = self.training_df.drop('label', axis=1)
        y = self.training_df['label']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Loaded {len(self.training_df)} training samples")
        print(f"   Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        
        # Load user's Drive activity for baseline
        self._learn_user_baseline()
        
        return True
    
    def _learn_user_baseline(self):
        """Learn user's normal Drive activity patterns."""
        print("\n📊 Learning your normal Drive behavior...")
        
        if not os.path.exists(EVENT_DB_PATH):
            print("   ⚠️ No user activity data yet")
            return
        
        try:
            cutoff = pd.Timestamp.now().timestamp() - (7 * 24 * 3600)
            
            with sqlite3.connect(EVENT_DB_PATH) as conn:
                query = f"SELECT * FROM events WHERE timestamp >= {cutoff}"
                self.user_events_df = pd.read_sql_query(query, conn)
            
            if self.user_events_df.empty:
                print("   ⚠️ No recent events")
                return
            
            # Calculate baseline metrics
            time_span = (self.user_events_df['timestamp'].max() - 
                        self.user_events_df['timestamp'].min())
            
            self.user_baseline['event_rate'] = len(self.user_events_df) / max(time_span, 1)
            
            # Find typical burst sizes
            self.user_events_df['minute'] = (self.user_events_df['timestamp'] // 60).astype(int)
            events_per_minute = self.user_events_df.groupby('minute').size()
            self.user_baseline['typical_burst_size'] = events_per_minute.quantile(0.95)
            
            # Common extensions
            extensions = self.user_events_df['local_title'].str.split('.').str[-1]
            self.user_baseline['common_extensions'] = extensions.value_counts().head(10).index.tolist()
            
            print(f"   ✅ Learned baseline:")
            print(f"      Event rate: {self.user_baseline['event_rate']:.4f} events/sec")
            print(f"      95th percentile burst: {self.user_baseline['typical_burst_size']:.0f} events/min")
            
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
    
    def create_optuna_study(self, study_name="drivetrail_config_optimization"):
        """Create or load Optuna study."""
        print(f"\n🔬 Creating Optuna study: '{study_name}'")
        
        # Use SQLite for persistence
        storage = f"sqlite:///{OPTUNA_DB_PATH}"
        
        # Create study with multi-objective optimization
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=["maximize", "maximize"],  # [F1-score, AUC]
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        return study
    
    def objective_likelihood_ratios(self, trial):
        """
        Objective function for optimizing Bayesian likelihood ratios.
        
        Tests different multiplier values and measures impact on detection.
        """
        # Suggest likelihood ratios
        high_velocity_lr = trial.suggest_float('high_velocity_lr', 1.5, 10.0)
        systematic_rename_lr = trial.suggest_float('systematic_rename_lr', 2.0, 15.0)
        
        # Simulate Bayesian fusion with these ratios
        # (Simplified - in reality we'd run through actual detection pipeline)
        
        # Use feature importance as proxy
        # High velocity correlates with File_created, file-related
        # Systematic rename correlates with extension_similarity, entropy
        
        hv_features = ['File_created', 'file-related']
        sr_features = ['extension_similarity', 'file_name_entropy']
        
        # Create weighted features
        X_train_weighted = self.X_train.copy()
        X_test_weighted = self.X_test.copy()
        
        # Apply weights
        for feat in hv_features:
            if feat in X_train_weighted.columns:
                X_train_weighted[feat] = X_train_weighted[feat] * high_velocity_lr
                X_test_weighted[feat] = X_test_weighted[feat] * high_velocity_lr
        
        for feat in sr_features:
            if feat in X_train_weighted.columns:
                X_train_weighted[feat] = X_train_weighted[feat] * systematic_rename_lr
                X_test_weighted[feat] = X_test_weighted[feat] * systematic_rename_lr
        
        # Train quick model
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train_weighted, self.y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_weighted)
        y_proba = model.predict_proba(X_test_weighted)[:, 1]
        
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        
        return f1, auc
    
    def objective_micropattern_thresholds(self, trial):
        """
        Objective function for micropattern detection thresholds.
        """
        # Suggest thresholds
        velocity_threshold = trial.suggest_int('velocity_threshold', 15, 100, step=5)
        rename_threshold = trial.suggest_int('rename_threshold', 3, 20)
        
        # Consider user baseline if available
        if self.user_baseline['typical_burst_size']:
            # Penalty if threshold is below user's normal activity
            user_burst = self.user_baseline['typical_burst_size']
            if velocity_threshold < user_burst:
                penalty = 0.1  # False positive penalty
            else:
                penalty = 0.0
        else:
            penalty = 0.0
        
        # Simulate detection on training data
        # Count how many benign samples would trigger (false positives)
        # Count how many malicious samples would trigger (true positives)
        
        benign = self.training_df[self.training_df['label'] == 0]
        malicious = self.training_df[self.training_df['label'] == 1]
        
        # Approximate: use File_created as proxy for velocity
        benign_velocity_triggers = (benign['File_created'] * 60 >= velocity_threshold).sum()
        malicious_velocity_triggers = (malicious['File_created'] * 60 >= velocity_threshold).sum()
        
        # Approximate: use extension_similarity for rename pattern
        benign_rename_triggers = (benign['extension_similarity'] < 0.5).sum() >= rename_threshold
        malicious_rename_triggers = (malicious['extension_similarity'] < 0.5).sum() >= rename_threshold
        
        # Calculate metrics
        tp = malicious_velocity_triggers + malicious_rename_triggers
        fp = benign_velocity_triggers + benign_rename_triggers
        fn = len(malicious) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Apply penalty
        f1 = f1 - penalty
        
        return f1, precision
    
    def objective_alert_thresholds(self, trial):
        """
        Objective function for alert probability thresholds.
        """
        # Suggest thresholds (must be in ascending order)
        alert_threshold = trial.suggest_float('alert_threshold', 0.30, 0.65)
        pause_threshold = trial.suggest_float('pause_threshold', alert_threshold + 0.05, 0.85)
        critical_threshold = trial.suggest_float('critical_threshold', pause_threshold + 0.05, 0.95)
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(self.X_train, self.y_train)
        
        # Get probabilities
        y_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate at pause threshold (most important)
        y_pred = (y_proba >= pause_threshold).astype(int)
        
        f1 = f1_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        
        # Bonus for logical threshold spacing
        spacing_bonus = (pause_threshold - alert_threshold) + (critical_threshold - pause_threshold)
        spacing_bonus = min(spacing_bonus / 0.4, 1.0) * 0.05  # Up to 5% bonus
        
        return f1 + spacing_bonus, recall
    
    def objective_hmm_parameters(self, trial):
        """
        Objective function for HMM parameters.
        """
        # Suggest parameters
        sequence_length = trial.suggest_int('sequence_length', 10, 35, step=5)
        n_sequences = trial.suggest_int('n_sequences', 300, 1500, step=100)
        
        # Evaluate using entropy distribution separation
        benign = self.training_df[self.training_df['label'] == 0]
        malicious = self.training_df[self.training_df['label'] == 1]
        
        benign_entropy = benign['file_name_entropy'].values
        malicious_entropy = malicious['file_name_entropy'].values
        
        # Create sequences
        benign_seqs = []
        malicious_seqs = []
        
        max_benign_seqs = min(n_sequences, len(benign_entropy) // sequence_length)
        max_malicious_seqs = min(n_sequences, len(malicious_entropy) // sequence_length)
        
        for i in range(max_benign_seqs):
            seq = benign_entropy[i*sequence_length:(i+1)*sequence_length]
            if len(seq) == sequence_length:
                benign_seqs.append(seq.mean())
        
        for i in range(max_malicious_seqs):
            seq = malicious_entropy[i*sequence_length:(i+1)*sequence_length]
            if len(seq) == sequence_length:
                malicious_seqs.append(seq.mean())
        
        if len(benign_seqs) == 0 or len(malicious_seqs) == 0:
            return 0.0, 0.0
        
        # Kolmogorov-Smirnov test for distribution separation
        from scipy import stats
        ks_stat, _ = stats.ks_2samp(benign_seqs, malicious_seqs)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(malicious_seqs) - np.mean(benign_seqs)
        pooled_std = np.sqrt((np.var(benign_seqs) + np.var(malicious_seqs)) / 2)
        cohens_d = mean_diff / (pooled_std + 1e-10)
        
        return ks_stat, cohens_d
    
    def optimize_all(self, n_trials=100):
        """
        Run comprehensive optimization for all parameters.
        """
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE OPTUNA OPTIMIZATION")
        print("="*70)
        
        results = {}
        
        # 1. Optimize Likelihood Ratios
        print("\n🎯 OPTIMIZING LIKELIHOOD RATIOS")
        print("-" * 70)
        study_lr = self.create_optuna_study("likelihood_ratios")
        study_lr.optimize(
            self.objective_likelihood_ratios,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_trial_lr = max(study_lr.best_trials, key=lambda t: t.values[0])  # Max F1
        results['likelihood_ratios'] = {
            'high_velocity': round(best_trial_lr.params['high_velocity_lr'], 2),
            'systematic_rename': round(best_trial_lr.params['systematic_rename_lr'], 2)
        }
        
        print(f"\n✅ Best Likelihood Ratios (F1: {best_trial_lr.values[0]:.4f}, AUC: {best_trial_lr.values[1]:.4f}):")
        print(f"   high_velocity: {results['likelihood_ratios']['high_velocity']}")
        print(f"   systematic_rename: {results['likelihood_ratios']['systematic_rename']}")
        
        # 2. Optimize Micropattern Thresholds
        print("\n🎯 OPTIMIZING MICROPATTERN THRESHOLDS")
        print("-" * 70)
        study_micro = self.create_optuna_study("micropattern_thresholds")
        study_micro.optimize(
            self.objective_micropattern_thresholds,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_trial_micro = max(study_micro.best_trials, key=lambda t: t.values[0])
        results['micropattern_thresholds'] = {
            'high_velocity': {
                'threshold': best_trial_micro.params['velocity_threshold'],
                'window_seconds': 60
            },
            'systematic_rename': {
                'threshold': best_trial_micro.params['rename_threshold'],
                'window_seconds': 60
            }
        }
        
        print(f"\n✅ Best Micropattern Thresholds (F1: {best_trial_micro.values[0]:.4f}):")
        print(f"   High Velocity: {results['micropattern_thresholds']['high_velocity']['threshold']} events/60s")
        print(f"   Systematic Rename: {results['micropattern_thresholds']['systematic_rename']['threshold']} files/60s")
        
        # 3. Optimize Alert Thresholds
        print("\n🎯 OPTIMIZING ALERT THRESHOLDS")
        print("-" * 70)
        study_alert = self.create_optuna_study("alert_thresholds")
        study_alert.optimize(
            self.objective_alert_thresholds,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_trial_alert = max(study_alert.best_trials, key=lambda t: t.values[0])
        results['alert_thresholds'] = {
            'alert': round(best_trial_alert.params['alert_threshold'], 2),
            'pause': round(best_trial_alert.params['pause_threshold'], 2),
            'critical': round(best_trial_alert.params['critical_threshold'], 2)
        }
        
        print(f"\n✅ Best Alert Thresholds (F1: {best_trial_alert.values[0]:.4f}):")
        print(f"   ALERT: {results['alert_thresholds']['alert']}")
        print(f"   PAUSE: {results['alert_thresholds']['pause']}")
        print(f"   CRITICAL: {results['alert_thresholds']['critical']}")
        
        # 4. Optimize HMM Parameters
        print("\n🎯 OPTIMIZING HMM PARAMETERS")
        print("-" * 70)
        study_hmm = self.create_optuna_study("hmm_parameters")
        study_hmm.optimize(
            self.objective_hmm_parameters,
            n_trials=n_trials // 2,  # Fewer trials needed
            show_progress_bar=True
        )
        
        best_trial_hmm = max(study_hmm.best_trials, key=lambda t: t.values[0])
        results['hmm_parameters'] = {
            'sequence_length': best_trial_hmm.params['sequence_length'],
            'n_states': 2,
            'n_sequences_per_class': best_trial_hmm.params['n_sequences']
        }
        
        print(f"\n✅ Best HMM Parameters (KS: {best_trial_hmm.values[0]:.4f}):")
        print(f"   Sequence Length: {results['hmm_parameters']['sequence_length']}")
        print(f"   Training Sequences: {results['hmm_parameters']['n_sequences_per_class']} per class")
        
        self.optimal_config = results
        
        # Generate visualizations
        self._generate_optuna_visualizations(study_lr, study_micro, study_alert, study_hmm)
        
        return results
    
    def _generate_optuna_visualizations(self, study_lr, study_micro, study_alert, study_hmm):
        """Generate Optuna visualization plots."""
        print("\n📊 Generating Optuna visualizations...")
        
        studies = [
            ('likelihood_ratios', study_lr),
            ('micropattern_thresholds', study_micro),
            ('alert_thresholds', study_alert),
            ('hmm_parameters', study_hmm)
        ]
        
        for name, study in studies:
            try:
                # Optimization history
                fig = plot_optimization_history(study)
                fig.write_image(os.path.join(ANALYSIS_OUTPUT_DIR, f'{name}_history.png'))
                
                # Parameter importances
                fig = plot_param_importances(study)
                fig.write_image(os.path.join(ANALYSIS_OUTPUT_DIR, f'{name}_importances.png'))
                
                # Parallel coordinate plot
                fig = plot_parallel_coordinate(study)
                fig.write_image(os.path.join(ANALYSIS_OUTPUT_DIR, f'{name}_parallel.png'))
                
                print(f"   ✅ Saved visualizations for {name}")
            except Exception as e:
                print(f"   ⚠️ Could not generate visualization for {name}: {e}")
        
        print(f"📁 All visualizations saved to: {ANALYSIS_OUTPUT_DIR}/")
    
    def save_config(self):
        """Save optimized configuration."""
        print(f"\n💾 Saving configuration to: {CONFIG_OUTPUT_PATH}")
        
        self.optimal_config['metadata'] = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'training_samples': len(self.training_df),
            'user_events': len(self.user_events_df) if self.user_events_df is not None else 0,
            'optimization_method': 'optuna_bayesian',
            'version': '2.0'
        }
        
        with open(CONFIG_OUTPUT_PATH, 'w') as f:
            json.dump(self.optimal_config, f, indent=2)
        
        print("✅ Configuration saved successfully!")
        print(f"\n📄 Config Preview:")
        print(json.dumps(self.optimal_config, indent=2))


def main():
    """Main optimization workflow."""
    print("="*70)
    print("  DRIVETRAIL OPTUNA-POWERED CONFIGURATION OPTIMIZER")
    print("="*70)
    print("\nUsing Bayesian optimization to find optimal configurations")
    print("This will take 5-15 minutes depending on your hardware.\n")
    
    optimizer = OptunaConfigOptimizer()
    
    if not optimizer.load_data():
        return
    
    # Run optimization
    n_trials = int(input("Number of trials per parameter (default 100): ") or "100")
    
    print(f"\n🚀 Starting optimization with {n_trials} trials per parameter...")
    print(f"   Total trials: ~{n_trials * 3.5:.0f}")
    print(f"   Estimated time: {n_trials * 3.5 / 60:.1f} minutes\n")
    
    results = optimizer.optimize_all(n_trials=n_trials)
    
    # Save configuration
    optimizer.save_config()
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\n📄 Configuration: {CONFIG_OUTPUT_PATH}")
    print(f"📊 Visualizations: {ANALYSIS_OUTPUT_DIR}/")
    print(f"💾 Optuna database: {OPTUNA_DB_PATH}")
    print("\nNext steps:")
    print("1. Review visualizations to understand optimization")
    print("2. Run: python scripts/apply_config.py")
    print("3. Retrain models and test!")


if __name__ == "__main__":
    main()