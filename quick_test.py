#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    colors = {"INFO": "\033[94m", "SUCCESS": "\033[92m", "WARNING": "\033[93m", "ERROR": "\033[91m"}
    reset = "\033[0m"
    color = colors.get(level, "")
    print(f"{color}[{timestamp}] {level}: {message}{reset}", flush=True)

def quick_data_test():
    log_message("Running quick data processing test...")
    
    try:
        from data.data_loader import F1DataLoader
        
        loader = F1DataLoader("data")
        races = ["Bahrain Grand Prix"]  # Just one race for quick test
        
        log_message(f"Processing test race: {races[0]}")
        data = loader.process_race_season(year=2024, races=races)
        
        if data and len(data['features']) > 0:
            log_message(f"✓ Data test passed: {len(data['features'])} samples", "SUCCESS")
            return True
        else:
            log_message("✗ No data processed", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"✗ Data test failed: {e}", "ERROR")
        return False

def quick_model_test():
    log_message("Running quick model test...")
    try:
        from models.transformer import create_model, MODEL_CONFIGS
        import torch
        
        config = MODEL_CONFIGS['small'].copy()
        config['input_dim'] = 19  # Standard F1 feature dimension
        
        model = create_model(config)
        
        dummy_input = torch.randn(2, 5, 19)  # batch_size=2, seq_len=5, features=19
        output = model(dummy_input)
        
        if output.shape[0] == 2:  # Check batch dimension
            log_message("✓ Model test passed", "SUCCESS")
            return True
        else:
            log_message("✗ Model output shape incorrect", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"✗ Model test failed: {e}", "ERROR")
        return False

def quick_training_test():
    log_message("Running quick training test...")
    
    data_path = "data/processed/f1_data_2024.pkl"
    if not Path(data_path).exists():
        log_message(f"✗ Data file not found: {data_path}", "ERROR")
        return False
    
    try:
        cmd = [
            "python", "training/train.py",
            "--data_path", data_path,
            "--model_size", "small",
            "--batch_size", "4",
            "--max_epochs", "1",
            "--mode", "transformer_only",
            "--no_wandb",
            "--fast_dev_run",
            "--output_dir", "test_quick_outputs"
        ]
        
        result = subprocess.run(cmd, text=True, timeout=120, capture_output=True)
        
        if result.returncode == 0:
            log_message("✓ Training test passed", "SUCCESS")
            return True
        else:
            log_message(f"✗ Training test failed: {result.stderr}", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        log_message("✗ Training test timed out", "ERROR")
        return False
    except Exception as e:
        log_message(f"✗ Training test failed: {e}", "ERROR")
        return False

def main():
    parser = argparse.ArgumentParser(description="Quick model testing utility")
    parser.add_argument("--test", choices=["data", "model", "training", "all"], 
                       default="all", help="Which test to run")
    args = parser.parse_args()
    
    log_message("Starting quick model tests...")
    
    tests = {
        "data": quick_data_test,
        "model": quick_model_test, 
        "training": quick_training_test
    }
    
    if args.test == "all":
        test_names = ["data", "model", "training"]
    else:
        test_names = [args.test]
    
    results = {}
    for test_name in test_names:
        results[test_name] = tests[test_name]()
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        log_message(f"🎉 All tests passed ({passed}/{total})", "SUCCESS")
    else:
        log_message(f"⚠️ {total-passed} test(s) failed ({passed}/{total})", "WARNING")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
