#!/bin/bash
#SBATCH --job-name=f1_2024_training
#SBATCH --partition=smallcard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-2024-%j.out
#SBATCH --error=slurm-2024-%j.err

set -euo pipefail

echo "F1 2024 Training Job - Running on: $(hostname)  JobID=$SLURM_JOB_ID  Date=$(date)"

nvidia-smi || echo "nvidia-smi failed (driver mismatch) - check system path"

export MKL_SERVICE_FORCE_INTEL=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.mplcache"
mkdir -p "$MPLCONFIGDIR"

set +u
eval "$(/mnt/slurm_nfs/a6abdulm/miniconda3/bin/conda shell.bash hook)"
conda activate f1-forecaster
set -u

python run_2024_training.py

echo "F1 2024 Training finished at $(date)"
