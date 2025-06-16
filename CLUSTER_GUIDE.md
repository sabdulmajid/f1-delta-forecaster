# F1 Tyre-Degradation Forecaster - Cluster Guide

## 🚀 Quick Start (Automated Pipeline)

The easiest way to run the complete pipeline is with the automated scripts:

### On ECE-NEBULA Cluster

1. **Clone and setup:**
```bash
ssh your_username@ece-nebula07.eng.uwaterloo.ca
cd /mnt/slurm_nfs/your_username
git clone <your-repo-url>
cd f1-delta-forecaster
bash cluster/setup_env.sh
```

2. **Run complete automation:**
```bash
sbatch cluster/slurm_job.sh
```

This will automatically:
- ✅ Test the pipeline with minimal data (5-10 minutes)
- ✅ Train full models on 2023-2024 data (2-4 hours)  
- ✅ Generate predictions for upcoming races (10-15 minutes)
- ✅ Save all results to `outputs/job_<SLURM_JOB_ID>/`

### Local Development

```bash
python run_f1_forecaster.py
```

This automatically detects your environment and runs the appropriate automation.

## 📊 Manual Steps (Optional)

If you want to run individual components:

1. **Setup environment:**
```bash
bash cluster/setup_env.sh
```

2. **Quick pipeline test:**
```bash
conda activate f1-forecaster
python test_pipeline.py
```

3. **Full training:**
```bash
python train_full.py
```

4. **Generate predictions:**
```bash
python predict_races.py
```

5. **Submit to SLURM:**
```bash
sbatch cluster/slurm_job.sh
```

6. **Monitor job:**
```bash
watch -n 1 squeue
```

7. **Check results:**
```bash
cat slurm-<job_id>.out
ls outputs/job_<job_id>/
```

## Local Development

1. **Create virtual environment:**
```bash
python3 -m venv f1-env
source f1-env/bin/activate
pip install -r requirements.txt
```

2. **Test data processing:**
```bash
python data/data_loader.py --year 2023 --races "Bahrain Grand Prix"
```

3. **Run training locally:**
```bash
python training/train.py --data_path data/processed/f1_data_2023.pkl --max_epochs 5
```

4. **Launch dashboard:**
```bash
streamlit run deployment/streamlit_app.py
```

## Transfer Results from Cluster

```bash
bash cluster/transfer_results.sh your_username <job_id>
```
