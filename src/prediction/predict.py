"""
Prediction and forecasting module.

This module handles loading trained models from MLflow,
generating forecasts, and visualizing predictions.
"""

import mlflow
import mlflow.pytorch
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, Optional
import seaborn as sns

from src.data.preprocessing import load_scaler, inverse_transform

logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")


def load_best_model(
    mlflow_run_id: str,
    mlflow_tracking_uri: str = './mlruns',
    device: torch.device = None
) -> torch.nn.Module:
    """
    Load the best model from MLflow.
    
    Args:
        mlflow_run_id: MLflow run ID
        mlflow_tracking_uri: MLflow tracking URI
        device: Device to load model on
        
    Returns:
        Loaded PyTorch model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Load model from MLflow
    model_uri = f"runs:/{mlflow_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    
    model.eval()
    logger.info(f"Loaded model from run {mlflow_run_id}")
    
    return model


def generate_forecast(
    model: torch.nn.Module,
    last_sequence: np.ndarray,
    horizon: int = 30,
    device: torch.device = None
) -> np.ndarray:
    """
    Generate multi-step forecast iteratively.
    
    Args:
        model: Trained LSTM model
        last_sequence: Last known sequence (sequence_length, n_features)
        horizon: Number of steps to forecast
        device: Device to run predictions on
        
    Returns:
        Array of predictions (horizon, n_features)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    # Convert to tensor and add batch dimension
    current_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
    
    predictions = []
    
    with torch.no_grad():
        for step in range(horizon):
            # Predict next step
            pred = model(current_seq)
            
            # Store prediction
            predictions.append(pred.cpu().numpy())
            
            # Update sequence: remove first step, add prediction
            pred_expanded = pred.unsqueeze(1)  # (1, 1, n_features)
            current_seq = torch.cat([current_seq[:, 1:, :], pred_expanded], dim=1)
            
            if (step + 1) % 10 == 0:
                logger.info(f"Generated {step + 1}/{horizon} predictions")
    
    # Concatenate predictions
    predictions = np.concatenate(predictions, axis=0)
    
    logger.info(f"Generated {horizon}-step forecast with shape {predictions.shape}")
    
    return predictions


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler_path: str
) -> np.ndarray:
    """
    Inverse transform normalized predictions back to original scale.
    
    Args:
        predictions: Normalized predictions
        scaler_path: Path to saved scaler
        
    Returns:
        Predictions in original scale
    """
    scaler = load_scaler(scaler_path)
    
    # Inverse transform
    original_scale = scaler.inverse_transform(predictions)
    
    logger.info(f"Inverse transformed predictions to original scale")
    logger.info(f"Prediction range: [{original_scale.min():.2f}, {original_scale.max():.2f}]")
    
    return original_scale


def plot_predictions(
    historical_data: pd.DataFrame,
    forecast_data: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    target_column: str = 'Close',
    save_path: Optional[str] = None,
    show_last_n_days: int = 180
) -> None:
    """
    Visualize historical data and forecasts.
    
    Args:
        historical_data: DataFrame with historical data
        forecast_data: Array of forecast values
        forecast_dates: DatetimeIndex for forecast period
        target_column: Name of the target column
        save_path: Optional path to save the plot
        show_last_n_days: Number of historical days to show
    """
    plt.figure(figsize=(14, 7))
    
    # Get last N days of historical data
    historical_subset = historical_data.tail(show_last_n_days)
    
    # Plot historical data
    plt.plot(
        historical_subset['Date'],
        historical_subset[target_column],
        label='Historical',
        color='blue',
        linewidth=2
    )
    
    # Plot forecast
    plt.plot(
        forecast_dates,
        forecast_data,
        label='Forecast',
        color='red',
        linewidth=2,
        linestyle='--'
    )
    
    # Mark the transition point
    plt.axvline(
        x=historical_data['Date'].iloc[-1],
        color='green',
        linestyle=':',
        linewidth=2,
        label='Forecast Start'
    )
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{target_column} Price ($)', fontsize=12)
    plt.title(f'NVIDIA Stock Price - Historical vs {len(forecast_data)}-Day Forecast',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction plot to {save_path}")
    
    plt.close()


def save_predictions_to_csv(
    forecast_data: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    save_path: str,
    column_names: list = None
) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        forecast_data: Array of forecast values
        forecast_dates: DatetimeIndex for forecast period
        save_path: Path to save CSV
        column_names: Optional list of column names
    """
    if column_names is None:
        if forecast_data.ndim == 1:
            column_names = ['Prediction']
        else:
            column_names = [f'Feature_{i}' for i in range(forecast_data.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(forecast_data, columns=column_names)
    df.insert(0, 'Date', forecast_dates)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    logger.info(f"Saved predictions to {save_path}")


def calculate_prediction_intervals(
    predictions: np.ndarray,
    confidence_level: float = 0.95,
    uncertainty_factor: float = 0.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals (simplified approach).
    
    Note: This is a simplified implementation using a fixed uncertainty factor.
    For more accurate intervals, consider using methods like bootstrapping,
    Monte Carlo dropout, or model ensembles.
    
    Args:
        predictions: Array of predictions
        confidence_level: Confidence level for intervals
        uncertainty_factor: Proportion of prediction used as uncertainty (default: 0.10 for 10%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    std = predictions * uncertainty_factor
    
    # Calculate z-score for confidence level
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    lower_bound = predictions - z_score * std
    upper_bound = predictions + z_score * std
    
    return lower_bound, upper_bound


def plot_predictions_with_intervals(
    historical_data: pd.DataFrame,
    forecast_data: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    target_column: str = 'Close',
    save_path: Optional[str] = None,
    show_last_n_days: int = 180
) -> None:
    """
    Visualize predictions with confidence intervals.
    
    Args:
        historical_data: DataFrame with historical data
        forecast_data: Array of forecast values
        forecast_dates: DatetimeIndex for forecast period
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        target_column: Name of the target column
        save_path: Optional path to save the plot
        show_last_n_days: Number of historical days to show
    """
    plt.figure(figsize=(14, 7))
    
    # Get last N days of historical data
    historical_subset = historical_data.tail(show_last_n_days)
    
    # Plot historical data
    plt.plot(
        historical_subset['Date'],
        historical_subset[target_column],
        label='Historical',
        color='blue',
        linewidth=2
    )
    
    # Plot forecast
    plt.plot(
        forecast_dates,
        forecast_data,
        label='Forecast',
        color='red',
        linewidth=2,
        linestyle='--'
    )
    
    # Plot confidence interval
    plt.fill_between(
        forecast_dates,
        lower_bound.flatten(),
        upper_bound.flatten(),
        alpha=0.3,
        color='red',
        label='95% Confidence Interval'
    )
    
    # Mark the transition point
    plt.axvline(
        x=historical_data['Date'].iloc[-1],
        color='green',
        linestyle=':',
        linewidth=2,
        label='Forecast Start'
    )
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{target_column} Price ($)', fontsize=12)
    plt.title(f'NVIDIA Stock Price - Historical vs {len(forecast_data)}-Day Forecast with Confidence Intervals',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction plot with intervals to {save_path}")
    
    plt.close()
