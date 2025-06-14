# Quick Start Guide for GPU Cluster

## Setup on ECE-NEBULA Cluster (University of Waterloo training cluster)

1. **Clone the repository:**
```bash
ssh your_username@ece-nebula07.eng.uwaterloo.ca
cd /mnt/slurm_nfs/your_username
git clone <your-repo-url>
cd f1-delta-forecaster
```

2. **Set up environment:**
```bash
bash cluster/setup_env.sh
```

3. **Process data (optional test):**
```bash
conda activate f1-forecaster
python data/data_loader.py --year 2023 --races "Bahrain Grand Prix" "Saudi Arabian Grand Prix"
```

4. **Submit training job:**
```bash
sbatch cluster/slurm_job.sh
```

5. **Monitor job:**
```bash
watch -n 1 squeue
```

6. **Check results:**
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
