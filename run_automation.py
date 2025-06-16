#!/usr/bin/env python3
"""
Master automation script: test -> train -> predict with error handling.
"""

import os
import sys
import time
import subprocess
import traceback
from pathlib import Path

def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}", flush=True)

def run_script(script_name, description, timeout_hours=1):
    log_message(f"Starting: {description}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["python", script_name],
            timeout=timeout_hours * 3600,
            text=True,
            capture_output=False
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log_message(f"✅ {description} completed in {elapsed/60:.1f} minutes")
            return True
        else:
            log_message(f"❌ {description} failed with exit code {result.returncode}", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        log_message(f"⏰ {description} timed out after {elapsed/3600:.1f} hours", "ERROR")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        log_message(f"💥 {description} crashed: {e}", "ERROR")
        traceback.print_exc()
        return False

def check_prerequisites():
    log_message("Checking prerequisites...")
    
    required_files = [
        "test_pipeline.py", "train_full.py", "predict_races.py",
        "data/data_loader.py", "training/train.py", "evaluation/metrics.py"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        log_message(f"Missing files: {missing_files}", "ERROR")
        return False
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'f1-forecaster':
        log_message(f"Warning: Not in f1-forecaster environment (current: {conda_env})", "WARNING")
    
    log_message("Prerequisites check passed")
    return True

def cleanup_previous_runs():
    log_message("Cleaning up previous runs...")
    
    import shutil
    cleanup_paths = ["test_outputs", "TRAINING_COMPLETE.txt", "predictions"]
    
    for path in cleanup_paths:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()
            log_message(f"Removed: {path}")

def save_run_summary(success_stages, failed_stages, total_time):
    summary = []
    summary.append("# F1 Forecaster Automation Summary")
    summary.append(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Total runtime: {total_time/3600:.2f} hours")
    summary.append("")
    
    summary.append("## Successful Stages")
    for stage in success_stages:
        summary.append(f"✅ {stage}")
    summary.append("")
    
    if failed_stages:
        summary.append("## Failed Stages") 
        for stage in failed_stages:
            summary.append(f"❌ {stage}")
        summary.append("")
    
    summary.append("## Generated Outputs")
    
    outputs = []
    if Path("data/processed").exists():
        data_files = list(Path("data/processed").glob("*.pkl"))
        outputs.append(f"- Data files: {len(data_files)} processed datasets")
    
    if Path("models/checkpoints").exists():
        model_files = list(Path("models/checkpoints").glob("**/*.ckpt"))
        outputs.append(f"- Model checkpoints: {len(model_files)} saved models")
    
    if Path("evaluation/results").exists():
        eval_files = list(Path("evaluation/results").glob("*"))
        outputs.append(f"- Evaluation results: {len(eval_files)} result files")
    
    if Path("predictions").exists():
        pred_files = list(Path("predictions").glob("*"))
        outputs.append(f"- Predictions: {len(pred_files)} prediction files")
    
    summary.extend(outputs)
    
    with open("AUTOMATION_SUMMARY.md", "w") as f:
        f.write("\n".join(summary))
    
    log_message("Run summary saved to AUTOMATION_SUMMARY.md")

def main():
    print("🏎️  F1 TYRE-DEGRADATION FORECASTER AUTOMATION")
    print("=" * 70)
    print("Pipeline: test -> train -> predict")
    print("=" * 70)
    
    start_time = time.time()
    success_stages = []
    failed_stages = []
    
    if not check_prerequisites():
        log_message("Prerequisites check failed. Aborting.", "ERROR")
        return False
    
    cleanup_previous_runs()
    
    # Stage 1: Quick Pipeline Test
    stage = "Pipeline Test"
    if run_script("test_pipeline.py", stage, timeout_hours=0.5):
        success_stages.append(stage)
        log_message("Pipeline test passed - proceeding to full training")
    else:
        failed_stages.append(stage)
        log_message("Pipeline test failed - aborting automation", "ERROR")
        save_run_summary(success_stages, failed_stages, time.time() - start_time)
        return False
    
    # Stage 2: Full Model Training
    stage = "Full Model Training"
    if run_script("train_full.py", stage, timeout_hours=6):
        success_stages.append(stage)
        log_message("Model training completed - proceeding to predictions")
    else:
        failed_stages.append(stage)
        log_message("Model training failed - skipping predictions", "ERROR")
        save_run_summary(success_stages, failed_stages, time.time() - start_time)
        return False
    
    # Stage 3: Race Predictions
    stage = "Race Predictions"
    if run_script("predict_races.py", stage, timeout_hours=0.5):
        success_stages.append(stage)
        log_message("Race predictions completed")
    else:
        failed_stages.append(stage)
        log_message("Race predictions failed", "ERROR")
    
    total_time = time.time() - start_time
    save_run_summary(success_stages, failed_stages, total_time)
    
    print("\n" + "=" * 70)
    if len(failed_stages) == 0:
        print("🎉 AUTOMATION COMPLETED SUCCESSFULLY!")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   All {len(success_stages)} stages completed")
    else:
        print("⚠️  AUTOMATION COMPLETED WITH ERRORS")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Successful: {len(success_stages)}, Failed: {len(failed_stages)}")
    
    print("\n📊 Generated outputs:")
    if Path("data/processed").exists():
        print("   - Training data: data/processed/")
    if Path("models/checkpoints").exists():
        print("   - Trained models: models/checkpoints/")  
    if Path("evaluation/results").exists():
        print("   - Evaluation: evaluation/results/")
    if Path("predictions").exists():
        print("   - Predictions: predictions/")
    
    print("\n📝 See AUTOMATION_SUMMARY.md for details")
    print("=" * 70)
    
    return len(failed_stages) == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log_message("Automation interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        log_message(f"Automation crashed: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)
