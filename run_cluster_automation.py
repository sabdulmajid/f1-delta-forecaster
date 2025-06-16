#!/usr/bin/env python3
"""
Cluster-optimized automation script for ECE-NEBULA.
Includes GPU checks, environment validation, and cluster-specific configurations.
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

def check_cluster_environment():
    """Check cluster-specific requirements."""
    log_message("Checking cluster environment...")
    
    # Check SLURM environment
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        log_message(f"Running in SLURM job: {job_id}")
        log_message(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
        log_message(f"CPUs: {os.environ.get('SLURM_CPUS_PER_TASK', 'unknown')}")
    else:
        log_message("Not in SLURM environment - running standalone")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            log_message(f"GPU available: {gpu_count} device(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                log_message(f"  GPU {i}: {gpu_name}")
        else:
            log_message("No GPU available - will use CPU", "WARNING")
    except ImportError:
        log_message("PyTorch not available", "ERROR")
        return False
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env == 'f1-forecaster':
        log_message(f"Conda environment: {conda_env}")
    else:
        log_message(f"Wrong conda environment: {conda_env} (expected: f1-forecaster)", "WARNING")
    
    return True

def setup_cluster_directories():
    """Set up cluster-specific directories and permissions."""
    log_message("Setting up cluster directories...")
    
    directories = [
        "data/processed",
        "models/checkpoints", 
        "evaluation/results",
        "predictions",
        f"outputs/job_{os.environ.get('SLURM_JOB_ID', 'local')}"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        log_message(f"Created directory: {directory}")

def run_with_gpu_optimization(script_name, description, timeout_hours=1):
    """Run script with GPU memory optimization."""
    log_message(f"Starting: {description}")
    
    # Set environment variables for GPU optimization
    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Prevent fragmentation
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["python", script_name],
            timeout=timeout_hours * 3600,
            env=env,
            text=True,
            capture_output=False  # Show output in real-time
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

def save_cluster_results():
    """Save results in cluster-appropriate format."""
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    output_dir = f"outputs/job_{job_id}"
    
    log_message(f"Saving results to {output_dir}...")
    
    # Copy important outputs
    copy_commands = []
    
    if Path("models/checkpoints").exists():
        copy_commands.append(f"cp -r models/checkpoints/* {output_dir}/")
    
    if Path("predictions").exists():
        copy_commands.append(f"cp -r predictions {output_dir}/")
    
    if Path("evaluation/results").exists():
        copy_commands.append(f"cp -r evaluation/results {output_dir}/")
    
    if Path("AUTOMATION_SUMMARY.md").exists():
        copy_commands.append(f"cp AUTOMATION_SUMMARY.md {output_dir}/")
    
    for cmd in copy_commands:
        try:
            subprocess.run(cmd, shell=True, check=True)
            log_message(f"Executed: {cmd}")
        except subprocess.CalledProcessError as e:
            log_message(f"Failed to copy: {cmd} - {e}", "WARNING")

def main():
    """Main cluster automation pipeline."""
    print("🏎️  F1 FORECASTER - CLUSTER AUTOMATION")
    print("=" * 70)
    print("Optimized for ECE-NEBULA GPU cluster")
    print("=" * 70)
    
    start_time = time.time()
    success_stages = []
    failed_stages = []
    
    # Cluster environment checks
    if not check_cluster_environment():
        log_message("Cluster environment check failed", "ERROR")
        return False
    
    # Setup directories
    setup_cluster_directories()
    
    # Stage 1: Quick Pipeline Test (reduced timeout for cluster)
    stage = "Pipeline Test"
    if run_with_gpu_optimization("test_pipeline.py", stage, timeout_hours=0.3):
        success_stages.append(stage)
        log_message("Pipeline test passed - proceeding to full training")
    else:
        failed_stages.append(stage)
        log_message("Pipeline test failed - aborting", "ERROR")
        return False
    
    # Stage 2: Full Model Training (longer timeout for cluster)
    stage = "Full Model Training" 
    if run_with_gpu_optimization("train_full.py", stage, timeout_hours=8):
        success_stages.append(stage)
        log_message("Model training completed - proceeding to predictions")
    else:
        failed_stages.append(stage)
        log_message("Model training failed - skipping predictions", "ERROR")
        save_cluster_results()
        return False
    
    # Stage 3: Race Predictions
    stage = "Race Predictions"
    if run_with_gpu_optimization("predict_races.py", stage, timeout_hours=0.3):
        success_stages.append(stage)
        log_message("Race predictions completed")
    else:
        failed_stages.append(stage)
        log_message("Race predictions failed", "ERROR")
    
    # Save all results
    save_cluster_results()
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    if len(failed_stages) == 0:
        print("🎉 CLUSTER AUTOMATION COMPLETED SUCCESSFULLY!")
        print(f"   Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   All {len(success_stages)} stages completed")
    else:
        print("⚠️  CLUSTER AUTOMATION COMPLETED WITH ERRORS")
        print(f"   Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Successful: {len(success_stages)}, Failed: {len(failed_stages)}")
    
    output_dir = f"outputs/job_{os.environ.get('SLURM_JOB_ID', 'local')}"
    print(f"\n📁 Results saved to: {output_dir}")
    print("📊 Generated outputs:")
    if Path("models/checkpoints").exists():
        print(f"   - Trained models: models/checkpoints/")
    if Path("predictions").exists():
        print(f"   - Predictions: predictions/")
    if Path("evaluation/results").exists():
        print(f"   - Evaluation: evaluation/results/")
    
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
