#!/usr/bin/env python3
"""
Universal F1 Forecaster Runner
Automatically detects environment (local/cluster) and runs appropriate automation.
"""

import os
import sys
import subprocess
from pathlib import Path

def detect_environment():
    """Detect if we're running on cluster or locally."""
    if os.environ.get('SLURM_JOB_ID'):
        return 'cluster'
    elif Path('/mnt/slurm_nfs').exists():
        return 'cluster'
    else:
        return 'local'

def main():
    env_type = detect_environment()
    
    print("🏎️  F1 TYRE-DEGRADATION FORECASTER")
    print("=" * 50)
    print(f"Environment detected: {env_type.upper()}")
    print("=" * 50)
    
    if env_type == 'cluster':
        print("Using cluster-optimized automation...")
        script = "run_cluster_automation.py"
    else:
        print("Using standard automation...")
        script = "run_automation.py"
    
    # Run the appropriate automation script
    try:
        result = subprocess.run([sys.executable, script], check=True)
        print(f"\n✅ Automation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Automation failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Automation interrupted by user")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
