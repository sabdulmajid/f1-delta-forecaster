#!/bin/bash
# Environment setup script for ECE-NEBULA cluster

echo "Setting up F1 Tyre-Degradation Forecaster environment..."

# Check if we're on the cluster
if [[ $HOSTNAME == *"ece-nebula"* ]]; then
    echo "Detected ECE-NEBULA cluster environment"
    CLUSTER_ENV=true
else
    echo "Local environment detected"
    CLUSTER_ENV=false
fi

# Install miniconda if not present (cluster environment)
if [ "$CLUSTER_ENV" = true ] && [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    source $HOME/miniconda3/etc/profile.d/conda.sh
else
    echo "Miniconda already installed or using local environment"
fi

# Initialize conda
if [ "$CLUSTER_ENV" = true ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
else
    # Try to find conda installation
    if command -v conda &> /dev/null; then
        echo "Conda found in PATH"
    else
        echo "Warning: Conda not found. Please install Anaconda/Miniconda first."
        exit 1
    fi
fi

# Create conda environment if it doesn't exist
if conda env list | grep -q "f1-forecaster"; then
    echo "Environment 'f1-forecaster' already exists"
else
    echo "Creating conda environment..."
    conda env create -f environment.yml
fi

# Activate environment
echo "Activating environment..."
conda activate f1-forecaster

# Install additional packages if needed
echo "Installing additional packages..."
pip install -r requirements.txt

# Verify GPU access (if available)
echo "Checking GPU access..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No GPU detected - will use CPU')
"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw/cache
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p outputs
mkdir -p evaluation/results

# Set up Fast-F1 cache directory
echo "Setting up Fast-F1 cache..."
python -c "
import fastf1
fastf1.Cache.enable_cache('data/raw/cache')
print('Fast-F1 cache configured')
"

# Download sample data if on cluster and data doesn't exist
if [ "$CLUSTER_ENV" = true ] && [ ! -f "data/processed/f1_data_2023.pkl" ]; then
    echo "Downloading and processing sample F1 data..."
    python data/data_loader.py --year 2023 --races "Bahrain Grand Prix" "Saudi Arabian Grand Prix" --data_dir data
fi

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate f1-forecaster"
echo ""
echo "To start training locally:"
echo "  python training/train.py --data_path data/processed/f1_data_2023.pkl"
echo ""
echo "To submit cluster job:"
echo "  sbatch cluster/slurm_job.sh"
