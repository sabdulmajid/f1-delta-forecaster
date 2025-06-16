#!/usr/bin/env python3
"""
Quick test to validate pipeline with minimal data.
"""

import os
import sys
import time
import traceback
from pathlib import Path

def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}", flush=True)

def test_data_processing():
    log_message("Testing data processing...")
    
    try:
        from data.data_loader import F1DataLoader
        
        loader = F1DataLoader("data")
        races = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]
        log_message(f"Processing races: {races}")
        
        data = loader.process_race_season(year=2024, races=races)
        
        log_message(f"Data processed successfully:")
        log_message(f"  Features: {data['features'].shape}")
        log_message(f"  Targets: {data['targets'].shape}")
        
        if len(data['features']) < 100:
            raise ValueError(f"Insufficient data: only {len(data['features'])} samples")
        
        return True
        
    except Exception as e:
        log_message(f"Data processing failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def test_model_training():
    log_message("Testing model training...")
    
    try:
        data_path = "data/processed/f1_data_2024.pkl"
        if not Path(data_path).exists():
            log_message(f"Data file not found: {data_path}", "ERROR")
            return False
        
        cmd = [
            "python", "training/train.py",
            "--data_path", data_path,
            "--model_size", "small",
            "--batch_size", "16", 
            "--learning_rate", "1e-3",
            "--max_epochs", "2",
            "--mode", "transformer_only",
            "--no_wandb",
            "--output_dir", "test_outputs"
        ]
        
        log_message(f"Running: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            log_message(f"Training failed with code {result.returncode}", "ERROR")
            log_message(f"STDERR: {result.stderr}", "ERROR")
            return False
        
        log_message("Model training test completed successfully")
        return True
        
    except Exception as e:
        log_message(f"Model training test failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def test_model_loading():
    log_message("Testing model loading...")
    
    try:
        checkpoint_dirs = [
            "test_outputs",
            "test_outputs/lightning_logs/version_0/checkpoints"
        ]
        
        checkpoint_path = None
        for dir_path in checkpoint_dirs:
            if Path(dir_path).exists():
                ckpt_files = list(Path(dir_path).glob("*.ckpt"))
                if ckpt_files:
                    checkpoint_path = str(ckpt_files[0])
                    break
        
        if not checkpoint_path:
            log_message("No checkpoint found for loading test", "ERROR")
            return False
        
        log_message(f"Loading checkpoint: {checkpoint_path}")
        from training.lightning_module import F1LightningModule
        model = F1LightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        log_message("Model loaded successfully")
        return True
        
    except Exception as e:
        log_message(f"Model loading test failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def main():
    log_message("Starting F1 Pipeline Quick Test")
    log_message("=" * 50)
    
    os.makedirs("test_outputs", exist_ok=True)
    
    if not test_data_processing():
        log_message("Data processing test failed - aborting", "ERROR")
        return False
    
    if not test_model_training():
        log_message("Model training test failed - aborting", "ERROR") 
        return False
    
    if not test_model_loading():
        log_message("Model loading test failed - aborting", "ERROR")
        return False
    
    log_message("=" * 50)
    log_message("✅ All tests passed! Pipeline is ready for full training.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
