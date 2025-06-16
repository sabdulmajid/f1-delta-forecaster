#!/usr/bin/env python3
"""
Comprehensive F1 model training with multiple years of data.
"""

import os
import sys
import time
import traceback
from pathlib import Path

def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}", flush=True)

def process_full_dataset():
    log_message("Processing multiple years of F1 data...")
    
    try:
        from data.data_loader import F1DataLoader
        
        loader = F1DataLoader("data")
        years_to_process = [2023, 2024]
        
        for year in years_to_process:
            log_message(f"Processing {year} season...")
            
            try:
                import fastf1
                schedule = fastf1.get_event_schedule(year)
                race_events = schedule[schedule['EventFormat'] != 'testing']
                races = race_events['EventName'].tolist()
                
                excluded_races = ['Pre-Season Test', 'Sprint Shootout', 'Sprint']
                races = [r for r in races if not any(excl in r for excl in excluded_races)]
                
                log_message(f"Processing {len(races)} races from {year}")
                
                data = loader.process_race_season(year=year, races=races)
                log_message(f"{year} season processed: {data['features'].shape[0]} samples")
                
            except Exception as e:
                log_message(f"Failed to process {year} season: {e}", "ERROR")
                continue
        
        data_files = list(Path("data/processed").glob("f1_data_*.pkl"))
        if not data_files:
            raise ValueError("No data files were created")
        
        log_message(f"Data processing completed. Files: {[f.name for f in data_files]}")
        return True
        
    except Exception as e:
        log_message(f"Data processing failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def train_full_model():
    log_message("Starting comprehensive model training...")
    
    try:
        data_files = list(Path("data/processed").glob("f1_data_*.pkl"))
        if not data_files:
            raise ValueError("No data files found")
        
        latest_file = sorted(data_files, key=lambda x: x.name)[-1]
        log_message(f"Training with dataset: {latest_file}")
        
        cmd = [
            "python", "training/train.py",
            "--data_path", str(latest_file),
            "--model_size", "medium",
            "--batch_size", "64",
            "--learning_rate", "1e-3", 
            "--max_epochs", "50",
            "--mode", "full",
            "--project_name", "f1-forecaster-full",
            "--output_dir", "models/checkpoints"
        ]
        
        log_message(f"Running: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, text=True, timeout=14400)
        
        if result.returncode != 0:
            log_message(f"Training failed with code {result.returncode}", "ERROR")
            return False
        
        log_message("Model training completed successfully")
        return True
        
    except Exception as e:
        log_message(f"Model training failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def evaluate_model():
    log_message("Starting model evaluation...")
    
    try:
        checkpoint_dirs = [
            "models/checkpoints/lightning_logs/version_0/checkpoints",
            "models/checkpoints"
        ]
        
        checkpoint_path = None
        for dir_path in checkpoint_dirs:
            if Path(dir_path).exists():
                ckpt_files = list(Path(dir_path).glob("*.ckpt"))
                if ckpt_files:
                    best_ckpts = [f for f in ckpt_files if 'last' not in f.name]
                    checkpoint_path = str(best_ckpts[0] if best_ckpts else ckpt_files[0])
                    break
        
        if not checkpoint_path:
            log_message("No checkpoint found for evaluation", "ERROR")
            return False
        
        data_files = list(Path("data/processed").glob("f1_data_*.pkl"))
        if not data_files:
            log_message("No data files found for evaluation", "ERROR")
            return False
        
        latest_data = sorted(data_files, key=lambda x: x.name)[-1]
        
        cmd = [
            "python", "evaluation/metrics.py",
            "--model_path", checkpoint_path,
            "--data_path", str(latest_data),
            "--output_dir", "evaluation/results"
        ]
        
        log_message(f"Running evaluation: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, text=True, timeout=1800)
        
        if result.returncode != 0:
            log_message(f"Evaluation failed with code {result.returncode}", "ERROR")
            return False
        
        log_message("Model evaluation completed successfully")
        return True
        
    except Exception as e:
        log_message(f"Model evaluation failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def main():
    log_message("Starting Comprehensive F1 Model Training")
    log_message("=" * 60)
    
    start_time = time.time()
    
    if not process_full_dataset():
        log_message("Data processing failed - aborting", "ERROR")
        return False
    
    if not train_full_model():
        log_message("Model training failed - aborting", "ERROR")
        return False
    
    if not evaluate_model():
        log_message("Model evaluation failed - continuing anyway", "WARNING")
    
    elapsed = time.time() - start_time
    log_message("=" * 60)
    log_message(f"✅ Full training pipeline completed in {elapsed/3600:.1f} hours")
    
    with open("TRAINING_COMPLETE.txt", "w") as f:
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {elapsed/3600:.1f} hours\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
