"""
PyTorch Lightning module for training the F1 tyre degradation forecaster.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from typing import Dict, Any, Optional, Tuple
import wandb

from models.transformer import create_model, MODEL_CONFIGS
from models.baseline import create_baseline_model, BASELINE_CONFIGS

class F1Dataset(Dataset):
    """Dataset class for F1 tyre degradation data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Input sequences of shape (n_samples, seq_len, n_features)
            targets: Target pace deltas of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class F1LightningModule(pl.LightningModule):
    """Lightning module for F1 tyre degradation forecasting."""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_config: Optional[Dict[str, Any]] = None,
        input_dim: Optional[int] = None
    ):
        """
        Initialize the Lightning module.
        
        Args:
            model_config: Configuration for the model
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            scheduler_config: Configuration for learning rate scheduler
            input_dim: Dimension of input features
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        
        # Initialize model
        if input_dim is None:
            raise ValueError("input_dim must be provided")
        
        self.model = create_model(input_dim=input_dim, **model_config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        features, targets = batch
        predictions = self(features).squeeze()
        loss = self.criterion(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', torch.mean(torch.abs(predictions - targets)), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        features, targets = batch
        predictions = self(features).squeeze()
        loss = self.criterion(predictions, targets)
        mae = torch.mean(torch.abs(predictions - targets))
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        features, targets = batch
        predictions = self(features).squeeze()
        loss = self.criterion(predictions, targets)
        mae = torch.mean(torch.abs(predictions - targets))
        rmse = torch.sqrt(loss)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        self.log('test_rmse', rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        features, _ = batch
        predictions = self(features).squeeze()
        return predictions
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if not self.scheduler_config:
            return optimizer
        
        scheduler_type = self.scheduler_config.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5),
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'frequency': 1
                }
            }
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 50),
                eta_min=self.scheduler_config.get('eta_min', 1e-6)
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

class F1DataModule(pl.LightningDataModule):
    """Lightning data module for F1 data."""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Initialize the data module.
        
        Args:
            data_path: Path to processed data file
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, validation, and test ratios must sum to 1.0"
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        import pickle
        
        # Load processed data
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        features = data['features']
        targets = data['targets']
        
        # Create full dataset
        full_dataset = F1Dataset(features, targets)
        
        # Calculate split sizes
        n_samples = len(full_dataset)
        train_size = int(self.train_ratio * n_samples)
        val_size = int(self.val_ratio * n_samples)
        test_size = n_samples - train_size - val_size
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Store input dimension for model initialization
        self.input_dim = features.shape[-1]
    
    def train_dataloader(self) -> DataLoader:
        """Training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Prediction data loader (uses test set)."""
        return self.test_dataloader()

def create_callbacks(config: Dict[str, Any]) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping', True):
        early_stop = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 10),
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop)
    
    # Model checkpointing
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.get('checkpoint_dir', 'models/checkpoints'),
        filename='f1-forecaster-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint)
    
    # Learning rate monitoring
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks

# Default configurations
DEFAULT_MODEL_CONFIG = MODEL_CONFIGS['medium'].copy()
DEFAULT_MODEL_CONFIG.update({
    'model_type': 'transformer'
})

DEFAULT_TRAINING_CONFIG = {
    'max_epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'early_stopping': True,
    'early_stopping_patience': 15,
    'checkpoint_dir': 'models/checkpoints',
    'scheduler_config': {
        'type': 'reduce_on_plateau',
        'factor': 0.5,
        'patience': 8
    }
}
