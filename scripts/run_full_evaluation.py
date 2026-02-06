"""
Master script to run complete evaluation pipeline for IEEE paper.
Executes all necessary steps in correct order.
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*70)
    print(f"  {description}")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_name}")
        print("   Please check the error messages above.")
        return False
    
    return True

def main():
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  DRIVETRAIL - COMPLETE EVALUATION PIPELINE".center(68) + "║")
    print("║" + "  (Automated execution for IEEE paper results)".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    scripts = [
        ("scripts/train_and_evaluate_enhanced.py", "Step 1: Training XGBoost with Comprehensive Metrics"),
        ("scripts/evaluate_baselines.py", "Step 2: Baseline Comparison Evaluation"),
        ("scripts/simulate_detection_latency.py", "Step 3: Detection Latency Simulation"),
        ("scripts/generate_paper_tables.py", "Step 4: Generating Paper-Ready Tables")
    ]
    
    print("\n📋 Pipeline Overview:")
    for i, (script, desc) in enumerate(scripts, 1):
        print(f"   {i}. {desc}")
    
    input("\n▶️  Press Enter to start the evaluation pipeline...")
    
    # Run all scripts in sequence
    for script, description in scripts:
        success = run_script(script, description)
        
        if not success:
            print("\n⚠️  Pipeline stopped due to error.")
            print(f"   Failed at: {description}")
            return
    
    # Final summary
    print("\n" + "="*70)
    print("  ✅ COMPLETE EVALUATION PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70)
    print("\n📊 Generated Results:")
    print("   • paper_results/xgboost_results.json")
    print("   • paper_results/baseline_comparison.json")
    print("   • paper_results/detection_latency_simulation.json")
    print("   • paper_results/confusion_matrix_data.json")
    print("\n📈 Generated Figures:")
    print("   • paper_results/figures/confusion_matrix.png")
    print("   • paper_results/figures/roc_curve.png")
    print("   • paper_results/figures/precision_recall_curve.png")
    print("   • paper_results/figures/feature_importance.png")
    print("\n📋 Generated Tables:")
    print("   • paper_results/paper_tables/*.csv")
    print("   • paper_results/paper_tables/*.tex")
    print("   • paper_results/paper_tables/RESULTS_SUMMARY_FOR_PAPER.txt")
    print("\n🎯 Next Steps:")
    print("   1. Review: paper_results/paper_tables/RESULTS_SUMMARY_FOR_PAPER.txt")
    print("   2. Copy metrics to your IEEE paper")
    print("   3. Insert figures from paper_results/figures/")
    print("   4. Use LaTeX tables from paper_results/paper_tables/")
    print("\n✨ Your paper is ready for results integration!")

if __name__ == "__main__":
    main()