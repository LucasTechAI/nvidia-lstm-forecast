"""
Training pipeline with MLflow integration.

This module handles model training, validation, early stopping,
and MLflow experiment tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # Ensure output shape matches target shape
        # This should not happen if model is configured correctly
        if outputs.shape != batch_y.shape:
            raise ValueError(
                f"Model output shape {outputs.shape} does not match target shape {batch_y.shape}. "
                f"Please check model output_size configuration."
            )
        
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            
            # Ensure output shape matches target shape
            if outputs.shape != batch_y.shape:
                raise ValueError(
                    f"Model output shape {outputs.shape} does not match target shape {batch_y.shape}. "
                    f"Please check model output_size configuration."
                )
            
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store for metric calculation
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    # Calculate metrics
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # MAPE (with threshold to handle near-zero values)
    # Only calculate MAPE for values above threshold
    threshold = 1e-3
    mask = np.abs(targets) > threshold
    if mask.any():
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = 0.0  # If all values near zero, set MAPE to 0
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    return avg_loss, metrics


def plot_training_history(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss plot to {save_path}")
    
    plt.close()


def train_model(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    config: Dict,
    device: torch.device,
    mlflow_tracking: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Complete training pipeline with MLflow tracking.
    
    Args:
        model: Neural network model
        train_data: Tuple of (X_train, y_train)
        val_data: Tuple of (X_val, y_val)
        config: Configuration dictionary with training parameters
        device: Device to train on
        mlflow_tracking: Whether to use MLflow tracking
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False
    )
    
    # Setup training
    criterion = nn.MSELoss()
    
    optimizer_name = config.get('optimizer', 'Adam')
    learning_rate = config.get('learning_rate', 0.001)
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training parameters
    epochs = config.get('epochs', 100)
    patience = config.get('early_stopping_patience', 10)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_mape': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Device: {device}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_mape'].append(val_metrics['mape'])
        
        # Log to MLflow
        if mlflow_tracking:
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_rmse', val_metrics['rmse'], step=epoch)
            mlflow.log_metric('val_mae', val_metrics['mae'], step=epoch)
            mlflow.log_metric('val_mape', val_metrics['mape'], step=epoch)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Val RMSE: {val_metrics['rmse']:.6f}"
            )
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with val_loss: {best_val_loss:.6f}")
    
    # Log final metrics
    if mlflow_tracking:
        mlflow.log_metric('best_val_loss', best_val_loss)
        mlflow.log_metric('training_time', training_time)
    
    return model, history


def save_model_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_model_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    device: torch.device
) -> Tuple[nn.Module, optim.Optimizer, int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, loss {loss:.6f})")
    
    return model, optimizer, epoch, loss
