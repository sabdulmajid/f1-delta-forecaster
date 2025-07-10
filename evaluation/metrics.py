"""
Model evaluation metrics and comparison utilities.
"""

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pickle
import argparse
from pathlib import Path
import json

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, fast_mode: bool = False) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics with optional fast mode.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        fast_mode: If True, compute only essential metrics for speed
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays and handle edge cases
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    # Basic metrics (always computed)
    metrics = {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    
    if not fast_mode:
        # Comprehensive metrics for detailed evaluation
        metrics.update({
            'mse': float(mean_squared_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
            'max_error': float(np.max(np.abs(y_true - y_pred))),
            'median_ae': float(np.median(np.abs(y_true - y_pred)))
        })
        
        # Additional F1-specific metrics
        metrics['accuracy_within_0.5s'] = float(np.mean(np.abs(y_true - y_pred) < 0.5) * 100)
    metrics['accuracy_within_1.0s'] = np.mean(np.abs(y_true - y_pred) < 1.0) * 100
    
    return metrics

def evaluate_transformer_model(
    model_path: str,
    data_path: str,
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    Evaluate a trained transformer model.
    
    Args:
        model_path: Path to model checkpoint
        data_path: Path to processed data
        device: Device to use for evaluation
    
    Returns:
        Evaluation results
    """
    # Load model
    from training.lightning_module import F1LightningModule
    model = F1LightningModule.load_from_checkpoint(model_path)
    model.eval()
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    targets = data['targets']
    
    # Split data (same as training)
    n_samples = len(features)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_test = features[train_size + val_size:]
    y_test = targets[train_size + val_size:]
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i + batch_size]
            batch_pred = model(batch).cpu().numpy().flatten()
            predictions.extend(batch_pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    
    return {
        'model_type': 'transformer',
        'model_path': model_path,
        'metrics': metrics,
        'predictions': predictions,
        'true_values': y_test,
        'test_size': len(y_test)
    }

def evaluate_baseline_models(
    baseline_results_path: str,
    data_path: str
) -> Dict[str, Any]:
    """
    Evaluate baseline models.
    
    Args:
        baseline_results_path: Path to baseline results
        data_path: Path to processed data
    
    Returns:
        Baseline evaluation results
    """
    # Load baseline models
    with open(baseline_results_path, 'rb') as f:
        baseline_results = pickle.load(f)
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    targets = data['targets']
    
    # Split data
    n_samples = len(features)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_test = features[train_size + val_size:]
    y_test = targets[train_size + val_size:]
    
    evaluation_results = {}
    
    for model_name, model_data in baseline_results.items():
        if 'model' in model_data:
            model = model_data['model']
            predictions = model.predict(X_test)
            metrics = calculate_metrics(y_test, predictions)
            
            evaluation_results[model_name] = {
                'model_type': 'baseline',
                'metrics': metrics,
                'predictions': predictions,
                'true_values': y_test,
                'test_size': len(y_test)
            }
    
    return evaluation_results

def create_comparison_plots(results: Dict[str, Any], output_dir: str = "evaluation/results"):
    """
    Create comparison plots for different models.
    
    Args:
        results: Dictionary of model evaluation results
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for comparison
    model_names = []
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    for model_name, model_results in results.items():
        model_names.append(model_name)
        mae_scores.append(model_results['metrics']['mae'])
        rmse_scores.append(model_results['metrics']['rmse'])
        r2_scores.append(model_results['metrics']['r2'])
    
    # Create comparison bar plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE comparison
    axes[0].bar(model_names, mae_scores)
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_ylabel('MAE (seconds)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # RMSE comparison
    axes[1].bar(model_names, rmse_scores)
    axes[1].set_title('Root Mean Square Error (RMSE)')
    axes[1].set_ylabel('RMSE (seconds)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[2].bar(model_names, r2_scores)
    axes[2].set_title('R² Score')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create prediction scatter plots
    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (model_name, model_results) in enumerate(results.items()):
        y_true = model_results['true_values']
        y_pred = model_results['predictions']
        
        # Scatter plot
        axes[i].scatter(y_true, y_pred, alpha=0.6)
        axes[i].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[i].set_xlabel('True Pace Delta (s)')
        axes[i].set_ylabel('Predicted Pace Delta (s)')
        axes[i].set_title(f'{model_name}\nMAE: {model_results["metrics"]["mae"]:.3f}s')
        axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create residual plots
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (model_name, model_results) in enumerate(results.items()):
        y_true = model_results['true_values']
        y_pred = model_results['predictions']
        residuals = y_true - y_pred
        
        # Residual plot
        axes[i].scatter(y_pred, residuals, alpha=0.6)
        axes[i].axhline(y=0, color='r', linestyle='--')
        axes[i].set_xlabel('Predicted Pace Delta (s)')
        axes[i].set_ylabel('Residuals (s)')
        axes[i].set_title(f'{model_name} Residuals')
        axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residual_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_evaluation_report(results: Dict[str, Any], output_dir: str = "evaluation/results"):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results: Dictionary of model evaluation results
        output_dir: Directory to save the report
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create metrics summary table
    metrics_df = pd.DataFrame({
        name: model_results['metrics'] 
        for name, model_results in results.items()
    }).T
    
    # Save metrics table
    metrics_df.to_csv(f"{output_dir}/metrics_summary.csv")
    
    # Generate markdown report
    report = []
    report.append("# F1 Tyre-Degradation Forecaster - Evaluation Report\n")
    
    report.append("## Model Comparison Summary\n")
    report.append(metrics_df.round(4).to_markdown())
    report.append("\n")
    
    # Best model analysis
    best_mae_model = metrics_df['mae'].idxmin()
    best_r2_model = metrics_df['r2'].idxmax()
    
    report.append("## Key Findings\n")
    report.append(f"- **Best MAE**: {best_mae_model} ({metrics_df.loc[best_mae_model, 'mae']:.4f}s)")
    report.append(f"- **Best R²**: {best_r2_model} ({metrics_df.loc[best_r2_model, 'r2']:.4f})")
    
    # Performance categorization
    for model_name, model_results in results.items():
        mae = model_results['metrics']['mae']
        acc_0_5 = model_results['metrics']['accuracy_within_0.5s']
        acc_1_0 = model_results['metrics']['accuracy_within_1.0s']
        
        report.append(f"\n### {model_name}")
        report.append(f"- MAE: {mae:.4f} seconds")
        report.append(f"- Accuracy within 0.5s: {acc_0_5:.1f}%")
        report.append(f"- Accuracy within 1.0s: {acc_1_0:.1f}%")
        
        if mae < 0.5:
            performance = "Excellent"
        elif mae < 1.0:
            performance = "Good"
        elif mae < 2.0:
            performance = "Fair"
        else:
            performance = "Needs Improvement"
        
        report.append(f"- **Performance**: {performance}")
    
    report.append("\n## Visualizations\n")
    report.append("- Model comparison: ![Model Comparison](model_comparison.png)")
    report.append("- Prediction accuracy: ![Prediction Scatter](prediction_scatter.png)")
    report.append("- Residual analysis: ![Residual Plots](residual_plots.png)")
    
    # Save report
    with open(f"{output_dir}/evaluation_report.md", 'w') as f:
        f.write('\n'.join(report))
    
    # Save detailed results as JSON
    json_results = {}
    for name, model_results in results.items():
        json_results[name] = {
            'model_type': model_results['model_type'],
            'metrics': model_results['metrics'],
            'test_size': model_results['test_size']
        }
    
    with open(f"{output_dir}/detailed_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)

# ----------------------------------------------------------------------
# Compatibility wrappers for training/train.py
# ----------------------------------------------------------------------
def evaluate_model(model, dataloader, device="cpu"):
    """
    Simple wrapper that runs the model on the dataloader and returns MAE/RMSE/R².
    You can replace this with something richer later.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x).detach()
            preds.append(out.cpu())
            targets.append(y.cpu())

    preds   = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()

    return calculate_metrics(targets, preds)            # reuse your helper above


def compare_models(metrics_a, metrics_b, key="mae"):
    """
    Return 'A' if metrics_a is better than metrics_b for the given key.
    Lower-is-better metrics assumed.
    """
    return "A" if metrics_a.get(key, float("inf")) < metrics_b.get(key, float("inf")) else "B"


def main():
    parser = argparse.ArgumentParser(description='Evaluate F1 forecasting models')
    parser.add_argument('--model_path', type=str, 
                       help='Path to transformer model checkpoint')
    parser.add_argument('--baseline_path', type=str,
                       help='Path to baseline results pickle file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data file')
    parser.add_argument('--output_dir', type=str, default='evaluation/results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    results = {}
    
    # Evaluate transformer model if provided
    if args.model_path and Path(args.model_path).exists():
        print("Evaluating transformer model...")
        transformer_results = evaluate_transformer_model(
            args.model_path, args.data_path, args.device
        )
        results['transformer'] = transformer_results
        print(f"Transformer MAE: {transformer_results['metrics']['mae']:.4f}")
    
    # Evaluate baseline models if provided
    if args.baseline_path and Path(args.baseline_path).exists():
        print("Evaluating baseline models...")
        baseline_results = evaluate_baseline_models(
            args.baseline_path, args.data_path
        )
        results.update(baseline_results)
        
        for name, model_results in baseline_results.items():
            print(f"{name} MAE: {model_results['metrics']['mae']:.4f}")
    
    if not results:
        print("No models found to evaluate. Please provide --model_path and/or --baseline_path")
        return
    
    # Generate visualizations and report
    print(f"Generating evaluation report in {args.output_dir}...")
    create_comparison_plots(results, args.output_dir)
    generate_evaluation_report(results, args.output_dir)
    
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
