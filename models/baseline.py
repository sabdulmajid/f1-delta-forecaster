"""
Baseline models for comparison with the transformer approach.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ExponentialSmoothingBaseline(BaseEstimator, RegressorMixin):
    """
    Exponential smoothing baseline for pace delta prediction.
    Simple but effective baseline for time-series forecasting.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize exponential smoothing model.
        
        Args:
            alpha: Smoothing parameter (0 < alpha < 1)
        """
        self.alpha = alpha
        self.last_values_ = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExponentialSmoothingBaseline':
        """
        Fit the exponential smoothing model.
        
        Args:
            X: Input sequences (not used for this baseline)
            y: Target pace deltas
        
        Returns:
            Self
        """
        # For this simple baseline, we just store the mean as initial value
        self.global_mean_ = np.mean(y)
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict pace deltas using exponential smoothing.
        
        Args:
            X: Input sequences
        
        Returns:
            Predicted pace deltas
        """
        if not hasattr(self, 'is_fitted_'):
            raise ValueError("Model must be fitted before making predictions")
        
        # Simple strategy: return global mean for all predictions
        # In a real implementation, this would use historical pace data
        return np.full(X.shape[0], self.global_mean_)

class MovingAverageBaseline(BaseEstimator, RegressorMixin):
    """
    Moving average baseline using recent lap times.
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize moving average model.
        
        Args:
            window_size: Number of previous values to average
        """
        self.window_size = window_size
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MovingAverageBaseline':
        """
        Fit the moving average model.
        
        Args:
            X: Input sequences 
            y: Target pace deltas
        
        Returns:
            Self
        """
        self.global_mean_ = np.mean(y)
        self.global_std_ = np.std(y)
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using moving average of recent values.
        
        Args:
            X: Input sequences
        
        Returns:
            Predicted pace deltas
        """
        if not hasattr(self, 'is_fitted_'):
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        
        for sequence in X:
            # Extract speed features from last few time steps
            recent_speeds = sequence[-self.window_size:, 0]  # Assuming first feature is speed
            speed_trend = np.mean(np.diff(recent_speeds)) if len(recent_speeds) > 1 else 0
            
            # Simple heuristic: slower speeds -> positive pace delta (slower lap)
            pred = speed_trend * -0.1 + np.random.normal(0, self.global_std_ * 0.1)
            predictions.append(pred)
        
        return np.array(predictions)

class LinearRegressionBaseline(BaseEstimator, RegressorMixin):
    """
    Linear regression baseline using aggregated features.
    """
    
    def __init__(self):
        self.model = LinearRegression()
    
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract aggregated features from sequences."""
        features = []
        
        for sequence in X:
            # Clean the sequence data
            sequence = np.array(sequence, dtype=np.float32)
            sequence = sequence[np.isfinite(sequence).all(axis=1)]

            if sequence.shape[0] < 2:
                # Not enough data to compute trends, use zeros
                # The feature vector size should be consistent.
                # mean, final, trend, stddev for 19 features = 19*4 = 76
                feature_vector = [0.0] * (19 * 4)
            else:
                feature_vector = []
                feature_vector.extend(np.mean(sequence, axis=0))
                feature_vector.extend(sequence[-1])
                feature_vector.extend(sequence[-1] - sequence[0])
                feature_vector.extend(np.std(sequence, axis=0))

            features.append(feature_vector)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionBaseline':
        """
        Fit the linear regression model.
        
        Args:
            X: Input sequences
            y: Target pace deltas
        
        Returns:
            Self
        """
        X_features = self._extract_features(X)
        self.model.fit(X_features, y)
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using linear regression.
        
        Args:
            X: Input sequences
        
        Returns:
            Predicted pace deltas
        """
        if not hasattr(self, 'is_fitted_'):
            raise ValueError("Model must be fitted before making predictions")
        
        X_features = self._extract_features(X)
        return self.model.predict(X_features)

class RandomForestBaseline(BaseEstimator, RegressorMixin):
    """
    Random Forest baseline using aggregated features.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract aggregated features from sequences."""
        features = []
        
        for sequence in X:
            # Clean the sequence data
            sequence = np.array(sequence, dtype=np.float32)
            sequence = sequence[np.isfinite(sequence).all(axis=1)]

            if sequence.shape[0] < 2:
                # Not enough data, use zeros.
                # mean, std, min, max, recent, trend, mean_diff = 19 * 7 = 133
                feature_vector = [0.0] * (19 * 7)
            else:
                feature_vector = []
                feature_vector.extend(np.mean(sequence, axis=0))
                feature_vector.extend(np.std(sequence, axis=0))
                feature_vector.extend(np.min(sequence, axis=0))
                feature_vector.extend(np.max(sequence, axis=0))
                feature_vector.extend(sequence[-1])
                feature_vector.extend(sequence[-1] - sequence[0])
                feature_vector.extend(np.mean(np.diff(sequence, axis=0), axis=0))

            features.append(feature_vector)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestBaseline':
        """
        Fit the random forest model.
        
        Args:
            X: Input sequences
            y: Target pace deltas
        
        Returns:
            Self
        """
        X_features = self._extract_features(X)
        self.model.fit(X_features, y)
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using random forest.
        
        Args:
            X: Input sequences
        
        Returns:
            Predicted pace deltas
        """
        if not hasattr(self, 'is_fitted_'):
            raise ValueError("Model must be fitted before making predictions")
        
        X_features = self._extract_features(X)
        return self.model.predict(X_features)

def create_baseline_model(model_type: str, **kwargs) -> BaseEstimator:
    """
    Factory function to create baseline models.
    
    Args:
        model_type: Type of baseline model
        **kwargs: Model parameters
    
    Returns:
        Initialized baseline model
    """
    if model_type == "exponential_smoothing":
        return ExponentialSmoothingBaseline(**kwargs)
    elif model_type == "moving_average":
        return MovingAverageBaseline(**kwargs)
    elif model_type == "linear_regression":
        return LinearRegressionBaseline(**kwargs)
    elif model_type == "random_forest":
        return RandomForestBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")

# Baseline model configurations
BASELINE_CONFIGS = {
    "exponential_smoothing": {
        "alpha": 0.3
    },
    "moving_average": {
        "window_size": 3
    },
    "linear_regression": {},
    "random_forest": {
        "n_estimators": 100,
        "random_state": 42
    }
}
