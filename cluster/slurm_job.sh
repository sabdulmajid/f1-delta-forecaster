#!/bin/bash
#SBATCH --job-name=f1-forecaster
#SBATCH --partition=midcard
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

# Process data if needed
if [ ! -f "data/processed/f1_data_2023.pkl" ]; then
    echo "Processing F1 data..."
    python data/data_loader.py --year 2023 --data_dir data
fi

# Train model
echo "Starting training..."
python training/train.py \
    --data_path data/processed/f1_data_2023.pkl \
    --model_size medium \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --max_epochs 50 \
    --mode full \
    --project_name f1-forecaster-cluster \
    --output_dir models/checkpoints

# Save results
mkdir -p outputs/job_${SLURM_JOB_ID}
cp -r models/checkpoints/* outputs/job_${SLURM_JOB_ID}/
cp slurm-${SLURM_JOB_ID}.out outputs/job_${SLURM_JOB_ID}/

echo "Job completed: $(date)"
