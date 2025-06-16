#!/bin/bash
#SBATCH --job-naecho "Running F1 Forecaster automation pipeline..."
python run_automation.py

mkdir -p outputs/job_${SLURM_JOB_ID}
[ -d "models/checkpoints" ] && cp -r models/checkpoints/* outputs/job_${SLURM_JOB_ID}/
[ -d "predictions" ] && cp -r predictions outputs/job_${SLURM_JOB_ID}/
[ -d "evaluation/results" ] && cp -r evaluation/results outputs/job_${SLURM_JOB_ID}/
cp AUTOMATION_SUMMARY.md outputs/job_${SLURM_JOB_ID}/ 2>/dev/null || true
cp slurm-${SLURM_JOB_ID}.out outputs/job_${SLURM_JOB_ID}/#SBATCH --partition=midcard
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "F1 Forecaster Training - Job $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

nvidia-smi

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate f1-forecaster

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the cluster-optimized automation pipeline
echo "Running F1 Forecaster cluster automation pipeline..."
python run_cluster_automation.py

# Save results
mkdir -p outputs/job_${SLURM_JOB_ID}
if [ -d "models/checkpoints" ]; then
    cp -r models/checkpoints/* outputs/job_${SLURM_JOB_ID}/
fi
if [ -d "predictions" ]; then
    cp -r predictions outputs/job_${SLURM_JOB_ID}/
fi
if [ -d "evaluation/results" ]; then
    cp -r evaluation/results outputs/job_${SLURM_JOB_ID}/
fi
cp AUTOMATION_SUMMARY.md outputs/job_${SLURM_JOB_ID}/ 2>/dev/null || true
cp slurm-${SLURM_JOB_ID}.out outputs/job_${SLURM_JOB_ID}/

echo "Job completed: $(date)"
