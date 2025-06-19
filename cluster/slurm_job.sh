#!/bin/bash
#SBATCH --job-name=f1_forecaster
#SBATCH --partition=midcard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Running on: $(hostname)  JobID=$SLURM_JOB_ID  Date=$(date)"

# Check NVIDIA driver version on compute node:
nvidia-smi || echo "nvidia-smi failed (driver mismatch) - check system path"

export MKL_SERVICE_FORCE_INTEL=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.mplcache"
mkdir -p "$MPLCONFIGDIR"

# Temporarily disable 'set -u' for conda activation
set +u
eval "$(/mnt/slurm_nfs/a6abdulm/miniconda3/bin/conda shell.bash hook)"
conda activate f1-forecaster
# Re-enable 'set -u' after conda activation
set -u

# Now, `set -e` is already active from the initial `set -euo pipefail`
# set -e # This line is redundant if `set -euo pipefail` is at the top

python run_cluster_automation.py

echo "Finished at $(date)"
