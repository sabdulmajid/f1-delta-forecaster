"""
Configuration settings for training and model parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class ModelConfig:
    """Configuration for transformer model."""
    input_dim: int
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 100
    model_type: str = "transformer"

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 4
    
    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    
    # Checkpointing
    checkpoint_dir: str = "models/checkpoints"
    save_top_k: int = 3
    
    # Logging
    use_wandb: bool = True
    project_name: str = "f1-tyre-degradation"
    
    # Scheduler
    scheduler_config: Optional[Dict[str, Any]] = None

@dataclass
class DataConfig:
    """Configuration for data processing."""
    year: int = 2023
    races: Optional[List[str]] = None
    sequence_length: int = 5
    cache_dir: str = "data/raw/cache"
    processed_dir: str = "data/processed"
    
    # Feature selection
    include_weather: bool = True
    include_telemetry: bool = True
    include_position: bool = True
    
    # Data filtering
    min_lap_time: float = 60.0  # seconds
    max_lap_time: float = 120.0  # seconds
    outlier_threshold: float = 3.0  # standard deviations

@dataclass
class ClusterConfig:
    """Configuration for cluster training."""
    partition: str = "midcard"
    cpus_per_task: int = 4
    memory_per_cpu: str = "12G"
    time_limit: str = "04:00:00"
    gpu_count: int = 1
    
    # Environment
    conda_env: str = "f1-forecaster"
    python_path: str = "python"
    
    # Paths
    slurm_output_dir: str = "outputs"
    checkpoint_dir: str = "models/checkpoints"

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig(
    input_dim=19,  # Will be updated based on actual data
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

DEFAULT_TRAINING_CONFIG = TrainingConfig(
    max_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    scheduler_config={
        'type': 'reduce_on_plateau',
        'factor': 0.5,
        'patience': 8
    }
)

DEFAULT_DATA_CONFIG = DataConfig(
    year=2023,
    sequence_length=5
)

DEFAULT_CLUSTER_CONFIG = ClusterConfig(
    partition="midcard",
    time_limit="04:00:00"
)

# Model size presets
MODEL_SIZE_CONFIGS = {
    "small": {
        "d_model": 256,
        "nhead": 4,
        "num_encoder_layers": 3,
        "dim_feedforward": 1024,
        "dropout": 0.1
    },
    "medium": {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1
    },
    "large": {
        "d_model": 768,
        "nhead": 12,
        "num_encoder_layers": 12,
        "dim_feedforward": 3072,
        "dropout": 0.1
    }
}

def get_config_for_cluster(partition: str = "midcard") -> ClusterConfig:
    """Get optimized configuration for specific cluster partition."""
    configs = {
        "smallcard": ClusterConfig(
            partition="smallcard",
            cpus_per_task=2,
            memory_per_cpu="8G",
            time_limit="02:00:00",
            gpu_count=1
        ),
        "midcard": ClusterConfig(
            partition="midcard",
            cpus_per_task=4,
            memory_per_cpu="12G",
            time_limit="04:00:00",
            gpu_count=1
        ),
        "dualcard": ClusterConfig(
            partition="dualcard",
            cpus_per_task=8,
            memory_per_cpu="16G",
            time_limit="08:00:00",
            gpu_count=2
        ),
        "bigcard": ClusterConfig(
            partition="bigcard",
            cpus_per_task=12,
            memory_per_cpu="20G",
            time_limit="12:00:00",
            gpu_count=2
        )
    }
    return configs.get(partition, DEFAULT_CLUSTER_CONFIG)
