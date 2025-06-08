#!/bin/bash
# Script to transfer training results from cluster to local machine

# Configuration
CLUSTER_USER="${1:-your_username}"
CLUSTER_HOST="ece-nebula07.eng.uwaterloo.ca"
CLUSTER_PATH="/mnt/slurm_nfs/${CLUSTER_USER}/f1-delta-forecaster"
LOCAL_PATH="./cluster_results"

echo "F1 Forecaster Results Transfer Script"
echo "====================================="

if [ $# -eq 0 ]; then
    echo "Usage: $0 <cluster_username> [job_id]"
    echo "Example: $0 ayman123 20356"
    echo ""
    echo "This script will download:"
    echo "  - Model checkpoints"
    echo "  - Training logs"
    echo "  - Evaluation results"
    echo "  - Job output files"
    exit 1
fi

JOB_ID="${2:-latest}"

# Create local results directory
mkdir -p "$LOCAL_PATH"

echo "Connecting to $CLUSTER_USER@$CLUSTER_HOST..."
echo "Downloading results to $LOCAL_PATH/"

# Download based on job ID
if [ "$JOB_ID" = "latest" ]; then
    echo "Downloading latest results..."
    
    # Get the most recent output directory
    LATEST_DIR=$(ssh "$CLUSTER_USER@$CLUSTER_HOST" "ls -1t $CLUSTER_PATH/outputs/ | head -1")
    
    if [ -z "$LATEST_DIR" ]; then
        echo "No output directories found on cluster"
        exit 1
    fi
    
    echo "Found latest directory: $LATEST_DIR"
    
    # Download the latest results
    scp -r "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/outputs/$LATEST_DIR" "$LOCAL_PATH/"
    
    # Also download model checkpoints
    scp -r "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/models/checkpoints" "$LOCAL_PATH/"
    
else
    echo "Downloading results for job $JOB_ID..."
    
    # Download specific job results
    scp -r "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/outputs/job_$JOB_ID" "$LOCAL_PATH/"
    
    # Download SLURM output files
    scp "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/slurm-$JOB_ID.out" "$LOCAL_PATH/"
    scp "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/slurm-$JOB_ID.err" "$LOCAL_PATH/"
fi

# Download processed data if not exists locally
if [ ! -f "data/processed/f1_data_2023.pkl" ]; then
    echo "Downloading processed data..."
    mkdir -p data/processed
    scp "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/data/processed/f1_data_2023.pkl" "data/processed/"
fi

# Download evaluation results if they exist
echo "Downloading evaluation results..."
scp -r "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/evaluation/results" "$LOCAL_PATH/" 2>/dev/null || echo "No evaluation results found"

echo ""
echo "Download complete!"
echo "Results saved to: $LOCAL_PATH/"

# Display summary
echo ""
echo "Files downloaded:"
find "$LOCAL_PATH" -type f -name "*.ckpt" -o -name "*.pkl" -o -name "*.json" -o -name "*.out" -o -name "*.err" | head -10

# Check for training summary
SUMMARY_FILE=$(find "$LOCAL_PATH" -name "training_summary.json" | head -1)
if [ -f "$SUMMARY_FILE" ]; then
    echo ""
    echo "Training Summary:"
    echo "=================="
    cat "$SUMMARY_FILE"
fi

echo ""
echo "To continue working with the results:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Run evaluation: python evaluation/metrics.py --model_path $LOCAL_PATH/checkpoints/last.ckpt"
echo "  3. Start Streamlit app: streamlit run deployment/streamlit_app.py"
