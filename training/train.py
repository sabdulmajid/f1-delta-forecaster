"""
Main training script for F1 tyre degradation forecaster.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import pickle
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.lightning_module import F1LightningModule, F1DataModule, create_callbacks
from training.lightning_module import DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG
from models.baseline import create_baseline_model, BASELINE_CONFIGS

def train_transformer_model(
    data_path: str,
    config: Dict[str, Any],
    use_wandb: bool = True,
    project_name: str = "f1-tyre-degradation"
) -> pl.LightningModule:
    """
    Train the transformer model.
    
    Args:
        data_path: Path to processed data
        config: Training configuration
        use_wandb: Whether to use Weights & Biases logging
        project_name: W&B project name
    
    Returns:
        Trained model
    """
    # Initialize W&B
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project=project_name,
            name=f"transformer-{config.get('model_size', 'medium')}",
            log_model=True
        )
    
    # Setup data module
    data_module = F1DataModule(
        data_path=data_path,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)
    )
    data_module.setup()
    
    # Initialize model
    model = F1LightningModule(
        model_config=config['model_config'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler_config=config.get('scheduler_config'),
        input_dim=data_module.input_dim
    )
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices='auto',
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=50
    )
    
    # Train model
    print(f"Starting training with {len(data_module.train_dataset)} training samples...")
    trainer.fit(model, data_module)
    
    # Test model
    print("Running final evaluation...")
    trainer.test(model, data_module)
    
    return model

def train_baseline_models(data_path: str) -> Dict[str, Any]:
    """
    Train baseline models for comparison.
    
    Args:
        data_path: Path to processed data
    
    Returns:
        Dictionary of trained baseline models
    """
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    targets = data['targets']
    
    # Split data (same as Lightning module)
    n_samples = len(features)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = features[:train_size]
    y_train = targets[:train_size]
    X_val = features[train_size:train_size + val_size]
    y_val = targets[train_size:train_size + val_size]
    X_test = features[train_size + val_size:]
    y_test = targets[train_size + val_size:]
    
    baseline_models = {}
    
    print("Training baseline models...")
    for model_name, model_config in BASELINE_CONFIGS.items():
        print(f"Training {model_name}...")
        
        model = create_baseline_model(model_name, **model_config)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mae = np.mean(np.abs(train_pred - y_train))
        val_mae = np.mean(np.abs(val_pred - y_val))
        test_mae = np.mean(np.abs(test_pred - y_test))
        
        print(f"{model_name} - Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}")
        
        baseline_models[model_name] = {
            'model': model,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae
        }
    
    return baseline_models

def main():
    parser = argparse.ArgumentParser(description='Train F1 tyre degradation forecaster')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to processed data file')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to training configuration file')
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size preset')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'transformer_only', 'baseline_only'],
                       help='Training mode')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default='f1-tyre-degradation',
                       help='W&B project name')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints',
                       help='Output directory for model checkpoints')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create configuration
    if args.config_path and os.path.exists(args.config_path):
        import json
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_TRAINING_CONFIG.copy()
        config['model_config'] = DEFAULT_MODEL_CONFIG.copy()
    
    # Override with command line arguments
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['max_epochs'] = args.max_epochs
    config['checkpoint_dir'] = args.output_dir
    
    # Update model config based on size
    from models.transformer import MODEL_CONFIGS
    if args.model_size in MODEL_CONFIGS:
        config['model_config'].update(MODEL_CONFIGS[args.model_size])
    
    print("Training Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Model size: {args.model_size}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Max epochs: {config['max_epochs']}")
    print(f"  Mode: {args.mode}")
    
    # Set random seeds for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Train models based on mode
    if args.mode in ['full', 'baseline_only']:
        print("\n" + "="*50)
        print("Training Baseline Models")
        print("="*50)
        baseline_results = train_baseline_models(args.data_path)
        
        # Save baseline results
        baseline_output_path = os.path.join(args.output_dir, 'baseline_results.pkl')
        with open(baseline_output_path, 'wb') as f:
            pickle.dump(baseline_results, f)
        print(f"Baseline results saved to {baseline_output_path}")
    
    if args.mode in ['full', 'transformer_only']:
        print("\n" + "="*50)
        print("Training Transformer Model")
        print("="*50)
        transformer_model = train_transformer_model(
            data_path=args.data_path,
            config=config,
            use_wandb=not args.no_wandb,
            project_name=args.project_name
        )
        
        print("Training completed!")
        print(f"Model checkpoints saved to {args.output_dir}")
    
    if args.mode == 'full':
        print("\n" + "="*50)
        print("Model Comparison Summary")
        print("="*50)
        
        # TODO: Add comprehensive model comparison
        print("Run evaluation/metrics.py for detailed comparison")

if __name__ == "__main__":
    import numpy as np  # Import here to avoid dependency issues during module import
    main()
