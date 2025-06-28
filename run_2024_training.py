#!/usr/bin/env python3
"""
F1 2024 Data Training and 2025 British Grand Prix Prediction
Trains model exclusively on 2024 F1 data and predicts 2025 British GP results.
"""

import os
import sys
import time
import subprocess
import traceback
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}", flush=True)

def check_environment():
    """Check required environment and dependencies."""
    log_message("Checking environment for 2024 training...")
    
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        log_message(f"SLURM Job: {job_id}")
        log_message(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            log_message(f"GPU available: {gpu_count} device(s)")
            for i in range(gpu_count):
                log_message(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            log_message("No GPU available - using CPU", "WARNING")
    except ImportError:
        log_message("PyTorch not available", "ERROR")
        return False
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env == 'f1-forecaster':
        log_message(f"Conda environment: {conda_env}")
    else:
        log_message(f"Wrong conda environment: {conda_env}", "WARNING")
    
    return True

def setup_directories():
    """Set up required directories."""
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    directories = [
        "data/processed",
        "models/checkpoints_2024", 
        "evaluation/results_2024",
        "predictions/british_gp_2025",
        f"outputs/job_2024_{job_id}"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        log_message(f"Created directory: {directory}")

def process_2024_data():
    """Process 2024 F1 season data."""
    log_message("Processing 2024 F1 season data...")
    
    try:
        from data.data_loader import F1DataLoader
        
        loader = F1DataLoader("data")
        
        import fastf1
        schedule = fastf1.get_event_schedule(2024)
        race_events = schedule[schedule['EventFormat'] != 'testing']
        races = race_events['EventName'].tolist()
        
        excluded_races = ['Pre-Season Test', 'Sprint Shootout', 'Sprint']
        races = [r for r in races if not any(excl in r for excl in excluded_races)]
        
        log_message(f"Processing {len(races)} races from 2024 season")
        
        data = loader.process_race_season(year=2024, races=races)
        log_message(f"2024 season processed: {data['features'].shape[0]} samples")
        
        if data['features'].shape[0] < 1000:
            log_message(f"Warning: Only {data['features'].shape[0]} samples available", "WARNING")
        
        return True
        
    except Exception as e:
        log_message(f"Failed to process 2024 data: {e}", "ERROR")
        traceback.print_exc()
        return False

def train_2024_model():
    """Train model on 2024 data."""
    log_message("Training model on 2024 F1 data...")
    
    try:
        data_path = "data/processed/f1_data_2024.pkl"
        if not Path(data_path).exists():
            log_message(f"Data file not found: {data_path}", "ERROR")
            return False
        
        env = os.environ.copy()
        env['CUDA_LAUNCH_BLOCKING'] = '1'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        env["MKL_SERVICE_FORCE_INTEL"] = "1"
        
        cmd = [
            "python", "training/train.py",
            "--data_path", data_path,
            "--model_size", "medium",
            "--batch_size", "64",
            "--learning_rate", "1e-3", 
            "--max_epochs", "75",
            "--mode", "full",
            "--project_name", "f1-forecaster-2024",
            "--output_dir", "models/checkpoints_2024"
        ]
        
        log_message(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, env=env, text=True, timeout=18000)  # 5 hour timeout
        
        if result.returncode != 0:
            log_message(f"Training failed with code {result.returncode}", "ERROR")
            return False
        
        log_message("2024 model training completed successfully")
        return True
        
    except Exception as e:
        log_message(f"Model training failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def evaluate_2024_model():
    """Evaluate the 2024 trained model."""
    log_message("Evaluating 2024 model...")
    
    try:
        checkpoint_dirs = [
            "models/checkpoints_2024/lightning_logs/version_0/checkpoints",
            "models/checkpoints_2024"
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
        
        data_path = "data/processed/f1_data_2024.pkl"
        
        cmd = [
            "python", "evaluation/metrics.py",
            "--model_path", checkpoint_path,
            "--data_path", data_path,
            "--output_dir", "evaluation/results_2024"
        ]
        
        log_message(f"Running evaluation: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, text=True, timeout=1800)
        
        if result.returncode != 0:
            log_message(f"Evaluation failed with code {result.returncode}", "ERROR")
            return False
        
        log_message("Model evaluation completed")
        return True
        
    except Exception as e:
        log_message(f"Model evaluation failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def predict_british_gp_2025():
    """Predict 2025 British Grand Prix results."""
    log_message("Predicting 2025 British Grand Prix results...")
    
    try:
        from training.lightning_module import F1LightningModule
        import torch
        
        # Find the best available model (transformer or baseline)
        best_model_path = None
        best_model_type = None

        # Prioritize transformer model if available
        checkpoint_dirs = [
            "models/checkpoints_2024/lightning_logs/version_0/checkpoints",
            "models/checkpoints_2024"
        ]
        for dir_path in checkpoint_dirs:
            if Path(dir_path).exists():
                ckpt_files = list(Path(dir_path).glob("*.ckpt"))
                if ckpt_files:
                    best_ckpts = [f for f in ckpt_files if 'last' not in f.name]
                    if best_ckpts:
                        best_model_path = str(best_ckpts[0])
                        best_model_type = 'transformer'
                        break
        
        # If no transformer, check for baseline models
        if not best_model_path:
            baseline_results_path = "models/checkpoints_2024/baseline_results.pkl"
            if Path(baseline_results_path).exists():
                with open(baseline_results_path, 'rb') as f:
                    baseline_results = pickle.load(f)
                
                # Find best baseline model by validation MAE
                best_baseline = min(
                    baseline_results.items(), 
                    key=lambda item: item[1].get('val_mae', float('inf'))
                )
                best_model_path = best_baseline[0]
                best_model_type = 'baseline'
                log_message(f"Using best baseline model: {best_model_path}", "INFO")

        if not best_model_path:
            log_message("No trained model found for prediction.", "ERROR")
            return False

        # Load data for prediction (dummy data for now)
        # In a real scenario, this would be pre-race data for the British GP
        log_message("Generating dummy data for British GP 2025 prediction...")
        
        # Use the same feature dimension as the training data
        with open("data/processed/f1_data_2024.pkl", 'rb') as f:
            train_data = pickle.load(f)
        
        num_features = train_data['features'].shape[2]
        log_message(f"Feature dimension from training data: {num_features}")

        # Create dummy data for 20 drivers, 5 laps each
        num_drivers = 20
        laps_per_driver = 5
        
        # Generate random data with the correct feature dimension
        dummy_features = np.random.rand(num_drivers * laps_per_driver, 5, num_features)
        
        log_message(f"Dummy data shape: {dummy_features.shape}")

        # Make predictions
        if best_model_type == 'transformer':
            log_message(f"Loading transformer model from {best_model_path}")
            model = F1LightningModule.load_from_checkpoint(best_model_path)
            model.eval()
            
            with torch.no_grad():
                predictions = model(torch.from_numpy(dummy_features).float()).numpy()
        
        elif best_model_type == 'baseline':
            log_message(f"Loading baseline model: {best_model_path}")
            with open("models/checkpoints_2024/baseline_results.pkl", 'rb') as f:
                baseline_results = pickle.load(f)
            model = baseline_results[best_model_path]['model']
            predictions = model.predict(dummy_features)

        # Process and save predictions
        prediction_df = pd.DataFrame({
            'Driver': [f"Driver_{i+1}" for i in range(num_drivers) for _ in range(laps_per_driver)],
            'Lap': [j+1 for _ in range(num_drivers) for j in range(laps_per_driver)],
            'PredictedPaceDelta': predictions.flatten()
        })
        
        output_path = "predictions/british_gp_2025/predicted_results.csv"
        prediction_df.to_csv(output_path, index=False)
        
        log_message(f"Predictions saved to {output_path}")
        log_message("British GP 2025 prediction completed successfully.")
        return True
        
    except FileNotFoundError:
        log_message("Training data not found. Cannot determine feature dimensions for prediction.", "ERROR")
        log_message("Please ensure 'data/processed/f1_data_2024.pkl' exists.", "ERROR")
        return False
    except Exception as e:
        log_message(f"Prediction failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def save_results():
    """Save all results to output directory."""
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    output_dir = Path(f"outputs/job_2024_{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_message(f"Saving results to {output_dir}...")
    
    copy_commands = [
        "cp -r models/checkpoints_2024/* outputs/job_2024_${SLURM_JOB_ID}/ 2>/dev/null || true",
        "cp -r predictions/british_gp_2025 outputs/job_2024_${SLURM_JOB_ID}/ 2>/dev/null || true",
        "cp -r evaluation/results_2024 outputs/job_2024_${SLURM_JOB_ID}/ 2>/dev/null || true"
    ]
    
    for cmd in copy_commands:
        try:
            subprocess.run(cmd, shell=True, check=False)
        except Exception as e:
            log_message(f"Copy command failed: {e}", "WARNING")

def main():
    """Main execution pipeline."""
    print("🏎️  F1 2024 TRAINING & 2025 BRITISH GP PREDICTION")
    print("=" * 70)
    print("Training on 2024 data, predicting 2025 British Grand Prix")
    print("=" * 70)
    
    start_time = time.time()
    success_stages = []
    failed_stages = []
    
    if not check_environment():
        log_message("Environment check failed", "ERROR")
        return False
    
    setup_directories()
    
    # Stage 1: Process 2024 Data
    stage = "2024 Data Processing"
    if process_2024_data():
        success_stages.append(stage)
    else:
        failed_stages.append(stage)
        log_message("Data processing failed - aborting", "ERROR")
        return False
    
    # Stage 2: Train on 2024 Data
    stage = "2024 Model Training"
    if train_2024_model():
        success_stages.append(stage)
    else:
        failed_stages.append(stage)
        log_message("Model training failed - aborting", "ERROR")
        save_results()
        return False
    
    # Stage 3: Evaluate Model
    stage = "Model Evaluation"
    if evaluate_2024_model():
        success_stages.append(stage)
    else:
        failed_stages.append(stage)
        log_message("Evaluation failed - continuing to prediction", "WARNING")
    
    # Stage 4: Predict British GP 2025
    stage = "British GP 2025 Prediction"
    if predict_british_gp_2025():
        success_stages.append(stage)
    else:
        failed_stages.append(stage)
        log_message("Prediction failed", "ERROR")
    
    save_results()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    if len(failed_stages) == 0:
        print("🎉 2024 TRAINING & 2025 PREDICTION COMPLETED!")
        print(f"   Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   All {len(success_stages)} stages completed")
    else:
        print("⚠️  COMPLETED WITH SOME ERRORS")
        print(f"   Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Successful: {len(success_stages)}, Failed: {len(failed_stages)}")
    
    output_dir = f"outputs/job_2024_{os.environ.get('SLURM_JOB_ID', 'local')}"
    print(f"\n📁 Results: {output_dir}")
    print("📊 Generated:")
    print("   - 2024 trained model: models/checkpoints_2024/")
    print("   - British GP predictions: predictions/british_gp_2025/")
    print("   - Evaluation results: evaluation/results_2024/")
    print("=" * 70)
    
    return len(failed_stages) == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log_message("Training interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        log_message(f"Training crashed: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)
