# F1 Tyre-Degradation Forecaster 🏎️

A state-of-the-art sequence-to-sequence transformer model for predicting Formula 1 lap pace deltas based on tyre compound, stint age, fuel load proxy, and recent micro-sector telemetry data.

## Project Overview

**Goal**: Predict a car's next-lap pace delta given compound, stint age, fuel load proxy, and recent micro-sectors.

**Why it's valuable**: Demonstrates mastery of multivariate time-series forecasting with attention mechanisms; provides actionable insights for broadcasters, teams, and strategy analysts.

## Key Features

- **Advanced ML Pipeline**: Transformer-based sequence-to-sequence architecture
- **Real-time Predictions**: GPU cluster deployment with SLURM job scheduling
- **Comprehensive Evaluation**: Multiple baseline models (exponential smoothing, moving average, linear regression, random forest)
- **Interactive Dashboard**: Streamlit web interface for race predictions
- **Production Ready**: Automated error handling, logging, and cluster deployment

## Implementation Pipeline

```
Fast-F1 → features per micro-sector
         (speed, throttle, nTurnsSinceStart, tyreAge, weather)
→ windowed dataset
→ Transformer-based seq2seq (PyTorch-Lightning)  
→ Evaluate MAE vs. simple exponential-smoothing baseline
→ Explainability: attention maps & SHAP
→ Streamlit widget: "slide tyre age" and watch forecast shift
```

## Project Structure

```
f1-delta-forecaster/
├── data/                     # Data processing and storage
│   ├── raw/                 # Raw F1 data from Fast-F1
│   ├── processed/           # Processed datasets
│   └── data_loader.py       # Data loading utilities
├── models/                   # Model architecture
│   ├── transformer.py       # Seq2seq transformer model
│   ├── baseline.py          # Exponential smoothing baseline
│   └── utils.py            # Model utilities
├── training/                 # Training scripts
│   ├── train.py            # Main training script
│   ├── config.py           # Training configuration
│   └── lightning_module.py  # PyTorch Lightning module
├── evaluation/              # Model evaluation
│   ├── metrics.py          # Evaluation metrics
│   ├── explainability.py   # SHAP and attention analysis
│   └── visualizations.py   # Plotting utilities
├── deployment/              # Deployment scripts
│   ├── streamlit_app.py    # Interactive Streamlit widget
│   └── inference.py        # Model inference utilities
├── cluster/                 # GPU cluster scripts
│   ├── slurm_job.sh        # SLURM job submission script
│   ├── setup_env.sh        # Environment setup script
│   └── transfer_results.sh  # Script to download results
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment specification
└── README.md               # This file
```

## Setup Instructions

### Local Development Setup

1. Clone the repository:
```bash
git clone 
cd f1-delta-forecaster
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate f1-forecaster
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

### GPU Cluster Setup (ECE-NEBULA)

1. Connect to the cluster (using the University of Waterloo engineering cluster for this project):
```bash
ssh <USERNAME>@ece-nebula07.eng.uwaterloo.ca
cd /mnt/slurm_nfs/<USERNAME>
```

2. Clone your repository:
```bash
git clone <your-repo-url>
cd f1-delta-forecaster
```

3. Set up the environment:
```bash
bash cluster/setup_env.sh
```

4. Submit training job:
```bash
sbatch cluster/slurm_job.sh
```

5. Monitor job:
```bash
watch -n 1 squeue
```

6. Download results:
```bash
bash cluster/transfer_results.sh
```

## Usage

### Data Preparation
```bash
python data/data_loader.py --year 2023 --races all
```

### Local Training (CPU/Small GPU)
```bash
python training/train.py --config training/config.py --mode local
```

### GPU Cluster Training
```bash
sbatch cluster/slurm_job.sh
```

### Evaluation
```bash
python evaluation/metrics.py --model_path models/checkpoints/best_model.ckpt
python evaluation/explainability.py --model_path models/checkpoints/best_model.ckpt
```

## Model Architecture

- **Input**: Sequence of micro-sector features (speed, throttle, turns, tyre age, weather)
- **Output**: Next-lap pace delta prediction
- **Architecture**: Transformer encoder-decoder with attention mechanisms
- **Framework**: PyTorch Lightning for scalable training

## Evaluation Metrics

- Mean Absolute Error (MAE) vs. exponential smoothing baseline
- Attention visualization for model interpretability
- SHAP values for feature importance analysis
