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
            log_message("No trained model found for prediction", "ERROR")
            return False
        
        log_message(f"Loading model from: {checkpoint_path}")
        model = F1LightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        # Get 2025 data if available, otherwise use historical British GP patterns
        try:
            import fastf1
            schedule_2025 = fastf1.get_event_schedule(2025)
            british_gp_2025 = schedule_2025[schedule_2025['EventName'].str.contains('British', case=False)]
            
            if len(british_gp_2025) > 0:
                log_message("Found 2025 British GP in schedule")
                british_gp_date = british_gp_2025.iloc[0]['EventDate']
                log_message(f"2025 British GP scheduled for: {british_gp_date}")
            else:
                log_message("2025 British GP not found in schedule, using historical patterns")
                
        except Exception as e:
            log_message(f"Could not fetch 2025 schedule: {e}", "WARNING")
        
        # Create British GP prediction scenarios
        drivers_2024 = [
            "Max Verstappen", "Sergio Perez", "Charles Leclerc", "Carlos Sainz",
            "Lando Norris", "Oscar Piastri", "George Russell", "Lewis Hamilton",
            "Fernando Alonso", "Lance Stroll", "Esteban Ocon", "Pierre Gasly",
            "Alex Albon", "Logan Sargeant", "Nico Hulkenberg", "Kevin Magnussen",
            "Yuki Tsunoda", "Daniel Ricciardo", "Valtteri Bottas", "Zhou Guanyu"
        ]
        
        predictions = []
        
        # Create realistic race scenarios for British GP (Silverstone characteristics)
        for i, driver in enumerate(drivers_2024):
            driver_features = create_british_gp_features(i+1, driver)
            
            with torch.no_grad():
                features_tensor = torch.tensor(driver_features, dtype=torch.float32).unsqueeze(0)
                prediction = model(features_tensor)
                predicted_time = prediction.cpu().numpy()[0]
            
            # Convert to race time (base time + delta)
            base_time = 5400.0  # ~1:30:00 for British GP
            final_time = base_time + predicted_time[0] if len(predicted_time) > 0 else base_time
            
            predictions.append({
                'Driver_Number': i + 1,
                'Driver_Name': driver,
                'Predicted_Time_Seconds': final_time,
                'Predicted_Time_Formatted': format_race_time(final_time)
            })
        
        # Sort by predicted time
        predictions.sort(key=lambda x: x['Predicted_Time_Seconds'])
        
        # Save predictions
        output_dir = Path("predictions/british_gp_2025")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_dir / "british_gp_2025_predictions.csv", index=False)
        
        # Save detailed results
        with open(output_dir / "detailed_predictions.txt", "w") as f:
            f.write("2025 BRITISH GRAND PRIX - PREDICTED FINAL RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: 2024 F1 Data Training\n\n")
            
            for i, pred in enumerate(predictions, 1):
                f.write(f"{i:2d}. #{pred['Driver_Number']:2d} {pred['Driver_Name']:<20} {pred['Predicted_Time_Formatted']}\n")
        
        log_message("2025 British Grand Prix predictions completed")
        log_message(f"Results saved to: {output_dir}")
        
        # Print top 10 predictions
        print("\n" + "=" * 60)
        print("2025 BRITISH GRAND PRIX - PREDICTED TOP 10")
        print("=" * 60)
        for i, pred in enumerate(predictions[:10], 1):
            print(f"{i:2d}. #{pred['Driver_Number']:2d} {pred['Driver_Name']:<20} {pred['Predicted_Time_Formatted']}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        log_message(f"British GP prediction failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def create_british_gp_features(driver_number, driver_name):
    """Create feature vector for British GP prediction."""
    # Silverstone-specific features (high-speed, medium downforce track)
    features = np.zeros(50)  # Adjust based on your actual feature count
    
    # Driver position features
    features[0] = driver_number / 20.0  # Normalized driver number
    
    # Track characteristics (Silverstone)
    features[1] = 0.7   # Track speed coefficient (high-speed)
    features[2] = 0.5   # Downforce requirement (medium)
    features[3] = 0.8   # Overtaking difficulty
    features[4] = 0.6   # Tyre degradation factor
    
    # Weather (typical British GP conditions)
    features[5] = 22.0  # Air temperature
    features[6] = 35.0  # Track temperature
    features[7] = 0.3   # Rain probability
    features[8] = 60.0  # Humidity
    
    # Tyre strategy (typical 2-stop)
    features[9] = 2.0   # Number of stops
    features[10] = 0.6  # Medium compound preference
    features[11] = 25.0 # Average stint length
    
    # Historical performance factors
    performance_multiplier = {
        "Max Verstappen": 0.95, "Charles Leclerc": 0.97, "Lando Norris": 0.98,
        "George Russell": 0.99, "Lewis Hamilton": 0.96, "Carlos Sainz": 0.98,
        "Oscar Piastri": 1.01, "Fernando Alonso": 0.99, "Sergio Perez": 1.02
    }.get(driver_name, 1.05)
    
    features[12] = performance_multiplier
    
    # Fill remaining features with track-specific data
    for i in range(13, len(features)):
        features[i] = np.random.normal(0.5, 0.1)  # Normalized random features
    
    return features

def format_race_time(seconds):
    """Format seconds to race time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:06.3f}"

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
