"""
Training Pipeline for NVIDIA LSTM Stock Prediction.

This module implements the complete training pipeline with:
- MLflow experiment tracking
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Comprehensive metrics logging
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import mlflow
import mlflow.pytorch
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import settings, TrainingConfig, MLflowConfig
from src.models.lstm_model import NvidiaLSTM, create_model
from src.etl.preprocessing import prepare_data_pipeline, inverse_transform

# Configure logging
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training when it stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopped
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None
        self.stopped = False
    
    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model being trained
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        self.counter += 1
        logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            self.stopped = True
            logger.info(f"Early stopping triggered at epoch {epoch}")
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info(f"Restored best weights from epoch {self.best_epoch}")
            return True
        
        return False


def get_optimizer(
    model: nn.Module,
    config: TrainingConfig
) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    optimizers = {
        'Adam': Adam,
        'AdamW': AdamW,
        'SGD': SGD,
        'RMSprop': RMSprop
    }
    
    optimizer_class = optimizers.get(config.optimizer, Adam)
    
    kwargs = {
        'lr': config.learning_rate,
        'weight_decay': config.weight_decay
    }
    
    if config.optimizer == 'SGD':
        kwargs['momentum'] = 0.9
    
    return optimizer_class(model.parameters(), **kwargs)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig
) -> Optional[Any]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        
    Returns:
        Configured scheduler or None
    """
    if not config.use_scheduler:
        return None
    
    if config.scheduler_type == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            min_lr=1e-7
        )
    elif config.scheduler_type == 'StepLR':
        return StepLR(optimizer, step_size=10, gamma=config.scheduler_factor)
    elif config.scheduler_type == 'CosineAnnealing':
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    return None


def get_loss_function(config: TrainingConfig) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Loss function
    """
    losses = {
        'MSE': nn.MSELoss(),
        'MAE': nn.L1Loss(),
        'Huber': nn.HuberLoss()
    }
    return losses.get(config.loss_function, nn.MSELoss())


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: Optional[float] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        gradient_clip: Max gradient norm for clipping
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for sequences, targets in dataloader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * len(sequences)
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Basic error metrics
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # MAPE (avoid division by zero)
    mask = all_targets != 0
    mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    
    # R² (coefficient of determination)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Correlation coefficient
    if len(all_preds) > 1:
        correlation = np.corrcoef(all_preds, all_targets)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Directional accuracy (% of correct direction predictions)
    if len(all_preds) > 1:
        pred_direction = np.sign(np.diff(all_preds))
        actual_direction = np.sign(np.diff(all_targets))
        directional_accuracy = np.mean(pred_direction == actual_direction) * 100
    else:
        directional_accuracy = 0.0
    
    # Max error
    max_error = np.max(np.abs(all_preds - all_targets))
    
    # === FINANCIAL/TRADING METRICS ===
    
    # MASE (Mean Absolute Scaled Error)
    if len(all_targets) > 1:
        naive_errors = np.abs(np.diff(all_targets))
        naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        mase = mae / naive_mae if naive_mae != 0 else float('inf')
    else:
        mase = 1.0
    
    # Theil's U statistic
    if len(all_targets) > 1:
        naive_preds = np.roll(all_targets, 1)[1:]
        naive_rmse = np.sqrt(np.mean((all_targets[1:] - naive_preds) ** 2))
        theils_u = rmse / naive_rmse if naive_rmse != 0 else float('inf')
    else:
        theils_u = 1.0
    
    # SMAPE
    smape = np.mean(2 * np.abs(all_preds - all_targets) / 
                    (np.abs(all_preds) + np.abs(all_targets) + 1e-8)) * 100
    
    # Trading metrics
    if len(all_preds) > 1:
        pred_returns = np.diff(all_preds) / (np.abs(all_preds[:-1]) + 1e-8)
        actual_returns = np.diff(all_targets) / (np.abs(all_targets[:-1]) + 1e-8)
        strategy_returns = np.sign(pred_returns) * actual_returns
        
        win_rate = np.mean(strategy_returns > 0) * 100
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0
        avg_loss_trade = np.abs(np.mean(strategy_returns[strategy_returns < 0])) if np.any(strategy_returns < 0) else 0
        win_loss_ratio = avg_win / avg_loss_trade if avg_loss_trade != 0 else float('inf')
        
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        win_loss_ratio = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
    
    return {
        'loss': avg_loss,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'r2': r2,
        'correlation': correlation,
        'directional_accuracy': directional_accuracy,
        'max_error': max_error,
        'mase': mase,
        'theils_u': theils_u,
        'win_rate': win_rate,
        'profit_factor': min(profit_factor, 100.0),
        'win_loss_ratio': min(win_loss_ratio, 100.0),
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item() * len(sequences)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Basic error metrics
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # MAPE (avoid division by zero)
    mask = all_targets != 0
    mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    
    # R² (coefficient of determination)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Correlation coefficient
    if len(all_preds) > 1:
        correlation = np.corrcoef(all_preds, all_targets)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Directional accuracy (% of correct direction predictions)
    if len(all_preds) > 1:
        pred_direction = np.sign(np.diff(all_preds))
        actual_direction = np.sign(np.diff(all_targets))
        directional_accuracy = np.mean(pred_direction == actual_direction) * 100
    else:
        directional_accuracy = 0.0
    
    # Max error
    max_error = np.max(np.abs(all_preds - all_targets))
    
    # === FINANCIAL/TRADING METRICS ===
    
    # MASE (Mean Absolute Scaled Error) - compares to naive forecast (previous value)
    # MASE < 1 means better than naive, MASE > 1 means worse
    if len(all_targets) > 1:
        naive_errors = np.abs(np.diff(all_targets))  # Error of naive forecast
        naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        mase = mae / naive_mae if naive_mae != 0 else float('inf')
    else:
        mase = 1.0
    
    # Theil's U statistic - ratio of RMSE to naive RMSE
    # U < 1 means better than naive, U > 1 means worse
    if len(all_targets) > 1:
        naive_preds = np.roll(all_targets, 1)[1:]  # Previous value as prediction
        naive_rmse = np.sqrt(np.mean((all_targets[1:] - naive_preds) ** 2))
        theils_u = rmse / naive_rmse if naive_rmse != 0 else float('inf')
    else:
        theils_u = 1.0
    
    # Symmetric MAPE (handles zero values better)
    smape = np.mean(2 * np.abs(all_preds - all_targets) / 
                    (np.abs(all_preds) + np.abs(all_targets) + 1e-8)) * 100
    
    # Profit/Loss simulation (if we traded based on predictions)
    if len(all_preds) > 1:
        # Predicted returns
        pred_returns = np.diff(all_preds) / (np.abs(all_preds[:-1]) + 1e-8)
        # Actual returns
        actual_returns = np.diff(all_targets) / (np.abs(all_targets[:-1]) + 1e-8)
        # If we bet on the predicted direction, what would be the return?
        strategy_returns = np.sign(pred_returns) * actual_returns
        
        # Win rate (% of profitable trades)
        win_rate = np.mean(strategy_returns > 0) * 100
        
        # Profit factor (gross profit / gross loss)
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Average win/loss ratio
        avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0
        avg_loss_trade = np.abs(np.mean(strategy_returns[strategy_returns < 0])) if np.any(strategy_returns < 0) else 0
        win_loss_ratio = avg_win / avg_loss_trade if avg_loss_trade != 0 else float('inf')
        
        # Sharpe ratio (annualized, assuming daily data)
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        win_loss_ratio = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
    
    return {
        'loss': avg_loss,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'r2': r2,
        'correlation': correlation,
        'directional_accuracy': directional_accuracy,
        'max_error': max_error,
        # Financial metrics
        'mase': mase,
        'theils_u': theils_u,
        'win_rate': win_rate,
        'profit_factor': min(profit_factor, 100.0),  # Cap at 100 to avoid inf
        'win_loss_ratio': min(win_loss_ratio, 100.0),
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_loss = min(val_losses)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best epoch: {best_epoch}')
    ax.scatter([best_epoch], [best_loss], color='g', s=100, zorder=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Loss curves saved to {save_path}")
    
    return fig


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    actuals: np.ndarray,
    scaler: Any,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot predictions vs actual values.
    
    Args:
        predictions: Predicted values (normalized)
        actuals: Actual values (normalized)
        scaler: Scaler for inverse transformation
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Inverse transform
    preds_orig = inverse_transform(predictions, scaler)
    actuals_orig = inverse_transform(actuals, scaler)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series plot
    ax1 = axes[0]
    ax1.plot(actuals_orig, 'b-', label='Actual', alpha=0.7)
    ax1.plot(preds_orig, 'r-', label='Predicted', alpha=0.7)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Stock Price ($)')
    ax1.set_title('Predictions vs Actual (Test Set)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(actuals_orig, preds_orig, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(actuals_orig.min(), preds_orig.min())
    max_val = max(actuals_orig.max(), preds_orig.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.set_title('Prediction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Prediction plot saved to {save_path}")
    
    return fig


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    model_config: Dict[str, Any],
    is_best: bool = False
) -> Path:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
        model_config: Model configuration
        is_best: Whether this is the best model so far
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model_config,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / 'latest_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model checkpoint to {best_path}")
    
    return checkpoint_path


def train_model(
    model: Optional[NvidiaLSTM] = None,
    dataloaders: Optional[Dict[str, DataLoader]] = None,
    scaler: Optional[Any] = None,
    training_config: Optional[TrainingConfig] = None,
    mlflow_config: Optional[MLflowConfig] = None,
    sequence_length: int = 60,
    run_name: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete model training pipeline with MLflow tracking.
    
    Args:
        model: LSTM model (created if None)
        dataloaders: Data loaders dict (created if None)
        scaler: Data scaler (obtained from data pipeline if None)
        training_config: Training configuration
        mlflow_config: MLflow configuration
        sequence_length: Sequence length for data preparation
        run_name: Name for MLflow run
        parent_run_id: Parent run ID for nested runs
        experiment_name: MLflow experiment name (uses mlflow_config if None)
        
    Returns:
        Dictionary with training results
    """
    if training_config is None:
        training_config = settings.training
    if mlflow_config is None:
        mlflow_config = settings.mlflow
    
    # Override experiment name if provided
    if experiment_name is not None:
        mlflow_config.experiment_name = experiment_name
    
    # Set random seeds for reproducibility
    torch.manual_seed(training_config.random_seed)
    np.random.seed(training_config.random_seed)
    
    # Device setup
    device = torch.device(settings.get_device())
    logger.info(f"Training on device: {device}")
    
    # Prepare data if not provided
    if dataloaders is None:
        dataloaders, scaler, df = prepare_data_pipeline(
            sequence_length=sequence_length,
            batch_size=training_config.batch_size
        )
    
    # Create model if not provided
    if model is None:
        model = create_model(device=str(device))
    else:
        model = model.to(device)
    
    # Setup training components
    optimizer = get_optimizer(model, training_config)
    scheduler = get_scheduler(optimizer, training_config)
    criterion = get_loss_function(training_config)
    early_stopping = EarlyStopping(
        patience=training_config.early_stopping_patience,
        min_delta=training_config.early_stopping_min_delta
    )
    
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(mlflow_config.experiment_name)
    
    # Generate run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{mlflow_config.run_name_prefix}_{timestamp}"
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name, nested=parent_run_id is not None):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Started MLflow run: {run_id}")
        
        # Log parameters
        mlflow.log_params({
            'sequence_length': sequence_length,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'dropout': model.dropout_prob,
            'bidirectional': model.bidirectional,
            'batch_size': training_config.batch_size,
            'learning_rate': training_config.learning_rate,
            'optimizer': training_config.optimizer,
            'loss_function': training_config.loss_function,
            'epochs': training_config.epochs,
            'early_stopping_patience': training_config.early_stopping_patience,
            'random_seed': training_config.random_seed,
            'device': str(device)
        })
        
        # Log model architecture
        mlflow.log_param('model_parameters', sum(p.numel() for p in model.parameters()))
        
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, training_config.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = train_epoch(
                model, dataloaders['train'], optimizer, criterion, device,
                gradient_clip=training_config.gradient_clip_value
            )
            
            # Validate
            val_metrics = validate_epoch(
                model, dataloaders['val'], criterion, device
            )
            
            # Update scheduler
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_rmse'].append(train_metrics['rmse'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['train_mae'].append(train_metrics['mae'])
            history['val_mae'].append(val_metrics['mae'])
            history['learning_rates'].append(current_lr)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_rmse': train_metrics['rmse'],
                'val_rmse': val_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'train_mape': train_metrics['mape'],
                'val_mape': val_metrics['mape'],
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2'],
                'train_correlation': train_metrics['correlation'],
                'val_correlation': val_metrics['correlation'],
                'train_directional_accuracy': train_metrics['directional_accuracy'],
                'val_directional_accuracy': val_metrics['directional_accuracy'],
                # Financial metrics
                'train_mase': train_metrics['mase'],
                'val_mase': val_metrics['mase'],
                'train_theils_u': train_metrics['theils_u'],
                'val_theils_u': val_metrics['theils_u'],
                'train_sharpe_ratio': train_metrics['sharpe_ratio'],
                'val_sharpe_ratio': val_metrics['sharpe_ratio'],
                'train_win_rate': train_metrics['win_rate'],
                'val_win_rate': val_metrics['win_rate'],
                'learning_rate': current_lr
            }, step=epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            logger.info(
                f"Epoch {epoch}/{training_config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Val RMSE: {val_metrics['rmse']:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['loss'],
                    training_config.checkpoint_dir,
                    model.get_config(),
                    is_best=True
                )
            
            # Early stopping check
            if early_stopping(val_metrics['loss'], model, epoch):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training complete in {total_time:.1f}s")
        logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        logger.info("=" * 60)
        
        # Test evaluation
        test_metrics = validate_epoch(model, dataloaders['test'], criterion, device)
        logger.info(f"Test metrics - Loss: {test_metrics['loss']:.6f}, "
                   f"RMSE: {test_metrics['rmse']:.6f}, MAE: {test_metrics['mae']:.6f}")
        
        mlflow.log_metrics({
            'test_loss': test_metrics['loss'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_mape': test_metrics['mape'],
            'test_smape': test_metrics['smape'],
            'test_r2': test_metrics['r2'],
            'test_correlation': test_metrics['correlation'],
            'test_directional_accuracy': test_metrics['directional_accuracy'],
            'test_max_error': test_metrics['max_error'],
            # Financial/Trading metrics
            'test_mase': test_metrics['mase'],
            'test_theils_u': test_metrics['theils_u'],
            'test_win_rate': test_metrics['win_rate'],
            'test_profit_factor': test_metrics['profit_factor'],
            'test_win_loss_ratio': test_metrics['win_loss_ratio'],
            'test_sharpe_ratio': test_metrics['sharpe_ratio'],
            'test_max_drawdown': test_metrics['max_drawdown'],
            'best_epoch': best_epoch,
            'total_training_time': total_time
        })
        
        # Save artifacts
        artifact_dir = settings.outputs_dir / 'artifacts' / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss curves
        loss_fig = plot_loss_curves(
            history['train_loss'],
            history['val_loss'],
            artifact_dir / 'loss_curves.png'
        )
        mlflow.log_artifact(str(artifact_dir / 'loss_curves.png'))
        plt.close(loss_fig)
        
        # Get test predictions for plotting
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for sequences, targets in dataloaders['test']:
                sequences = sequences.to(device)
                preds = model(sequences)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        pred_fig = plot_predictions_vs_actual(
            np.array(all_preds),
            np.array(all_targets),
            scaler,
            artifact_dir / 'predictions_vs_actual.png'
        )
        mlflow.log_artifact(str(artifact_dir / 'predictions_vs_actual.png'))
        plt.close(pred_fig)
        
        # Save scaler
        import joblib
        scaler_path = artifact_dir / 'scaler.joblib'
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path))
        
        # Log model
        if mlflow_config.log_models:
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name=mlflow_config.registered_model_name
            )
        
        results = {
            'run_id': run_id,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'history': history,
            'training_time': total_time,
            'model': model,
            'scaler': scaler
        }
        
        return results


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NVIDIA LSTM model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length')
    parser.add_argument('--hidden-size', type=int, default=None, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=None, help='Number of LSTM layers')
    parser.add_argument('--run-name', type=str, default=None, help='MLflow run name')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format
    )
    
    # Override config with command line args
    training_config = settings.training
    if args.epochs:
        training_config.epochs = args.epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    
    lstm_config = settings.lstm
    if args.hidden_size:
        lstm_config.hidden_size = args.hidden_size
    if args.num_layers:
        lstm_config.num_layers = args.num_layers
    
    # Create model with updated config
    model = create_model(config=lstm_config)
    
    # Train
    results = train_model(
        model=model,
        training_config=training_config,
        sequence_length=args.sequence_length,
        run_name=args.run_name
    )
    
    print(f"\nTraining complete!")
    print(f"Run ID: {results['run_id']}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.6f}")


if __name__ == '__main__':
    main()
