"""
Prediction Module for NVIDIA LSTM Stock Forecasting.

This module implements:
- Loading trained models from MLflow
- Generating multi-step forecasts
- Inverse transforming predictions
- Visualization of forecasts with confidence intervals
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import joblib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import settings, PredictionConfig
from src.models.lstm_model import NvidiaLSTM, load_model_from_checkpoint
from etl.preprocessing import (
    load_data_from_db,
    normalize_features,
    get_last_sequence,
    inverse_transform
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast results."""
    
    # Forecast data
    predictions: np.ndarray  # Predicted values (original scale)
    dates: List[datetime]  # Forecast dates
    
    # Historical context
    historical_prices: np.ndarray
    historical_dates: List[datetime]
    
    # Uncertainty (if available)
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    
    # Metadata
    model_run_id: Optional[str] = None
    forecast_horizon: int = 30
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to DataFrame."""
        df = pd.DataFrame({
            'Date': self.dates,
            'Predicted_Price': self.predictions
        })
        
        if self.lower_bound is not None:
            df['Lower_Bound'] = self.lower_bound
        if self.upper_bound is not None:
            df['Upper_Bound'] = self.upper_bound
            
        return df
    
    def save(self, path: Path, format: str = 'csv') -> None:
        """Save forecast to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        
        if format == 'csv':
            df.to_csv(path, index=False)
        elif format == 'json':
            df.to_json(path, orient='records', date_format='iso')
        elif format == 'parquet':
            df.to_parquet(path, index=False)
        
        logger.info(f"Forecast saved to {path}")


def load_best_model(
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[NvidiaLSTM, Dict[str, Any]]:
    """
    Load the best trained model.
    
    Supports loading from:
    1. MLflow run ID
    2. MLflow model registry
    3. Local checkpoint file
    
    Args:
        run_id: MLflow run ID to load model from
        model_name: Name of registered model in MLflow
        checkpoint_path: Path to local checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (model, metadata)
    """
    if device is None:
        device = settings.get_device()
    
    metadata = {}
    
    if checkpoint_path:
        # Load from local checkpoint
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, device)
        metadata['source'] = 'checkpoint'
        metadata['checkpoint_path'] = checkpoint_path
        
    elif run_id:
        # Load from MLflow run
        logger.info(f"Loading model from MLflow run: {run_id}")
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        
        # Get run info
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        metadata = {
            'source': 'mlflow_run',
            'run_id': run_id,
            'run_name': run.info.run_name,
            'experiment_id': run.info.experiment_id,
            'params': run.data.params,
            'metrics': run.data.metrics
        }
        
    elif model_name:
        # Load from model registry
        logger.info(f"Loading model from registry: {model_name}")
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        metadata['source'] = 'model_registry'
        metadata['model_name'] = model_name
        
    else:
        # Try to load from default checkpoint
        default_checkpoint = settings.training.checkpoint_dir / 'best_model.pt'
        if default_checkpoint.exists():
            logger.info(f"Loading from default checkpoint: {default_checkpoint}")
            model = load_model_from_checkpoint(str(default_checkpoint), device)
            metadata['source'] = 'default_checkpoint'
        else:
            raise ValueError("No model source specified and no default checkpoint found")
    
    model.eval()
    logger.info(f"Model loaded successfully to {device}")
    
    return model, metadata


def load_scaler(
    run_id: Optional[str] = None,
    scaler_path: Optional[str] = None
) -> Any:
    """
    Load the scaler used during training.
    
    Args:
        run_id: MLflow run ID to load scaler from
        scaler_path: Direct path to scaler file
        
    Returns:
        Fitted scaler object
    """
    if scaler_path:
        logger.info(f"Loading scaler from: {scaler_path}")
        return joblib.load(scaler_path)
    
    if run_id:
        logger.info(f"Loading scaler from MLflow run: {run_id}")
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        
        # Download artifacts
        client = mlflow.tracking.MlflowClient()
        artifact_path = client.download_artifacts(run_id, "scaler.joblib")
        return joblib.load(artifact_path)
    
    # Try default location
    default_paths = [
        settings.outputs_dir / 'artifacts' / 'scaler.joblib',
        settings.models_dir / 'scaler.joblib'
    ]
    
    for path in default_paths:
        if path.exists():
            logger.info(f"Loading scaler from: {path}")
            return joblib.load(path)
    
    raise FileNotFoundError("Could not find scaler file")


def generate_forecast(
    model: NvidiaLSTM,
    last_sequence: torch.Tensor,
    horizon: int = 30,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Generate multi-step forecast.
    
    Uses iterative prediction where each predicted value is appended
    to the sequence for the next prediction.
    
    Args:
        model: Trained LSTM model
        last_sequence: Last known sequence (1, seq_len, n_features)
        horizon: Number of time steps to predict
        device: Device to run on
        
    Returns:
        Array of predictions (horizon,)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    predictions = []
    
    current_sequence = last_sequence.clone().to(device)
    
    logger.info(f"Generating {horizon}-step forecast")
    
    with torch.no_grad():
        for step in range(horizon):
            # Predict next value
            pred = model(current_sequence)
            predictions.append(pred.item())
            
            # Update sequence: slide window and add prediction
            new_input = pred.view(1, 1, -1)
            current_sequence = torch.cat([
                current_sequence[:, 1:, :],
                new_input
            ], dim=1)
    
    return np.array(predictions)


def generate_forecast_with_uncertainty(
    model: NvidiaLSTM,
    last_sequence: torch.Tensor,
    horizon: int = 30,
    n_samples: int = 100,
    dropout_rate: float = 0.1,
    confidence_level: float = 0.95,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate forecast with uncertainty estimation using Monte Carlo Dropout.
    
    Args:
        model: Trained LSTM model
        last_sequence: Last known sequence
        horizon: Number of time steps to predict
        n_samples: Number of Monte Carlo samples
        dropout_rate: Dropout rate for uncertainty estimation
        confidence_level: Confidence level for intervals
        device: Device to run on
        
    Returns:
        Tuple of (mean_predictions, lower_bound, upper_bound)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Enable dropout for uncertainty estimation
    model.train()  # Keep dropout active
    
    all_predictions = []
    
    logger.info(f"Generating {horizon}-step forecast with {n_samples} MC samples")
    
    for _ in range(n_samples):
        current_sequence = last_sequence.clone().to(device)
        sample_predictions = []
        
        with torch.no_grad():
            for step in range(horizon):
                pred = model(current_sequence)
                sample_predictions.append(pred.item())
                
                new_input = pred.view(1, 1, -1)
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    new_input
                ], dim=1)
        
        all_predictions.append(sample_predictions)
    
    model.eval()  # Restore eval mode
    
    all_predictions = np.array(all_predictions)
    
    # Calculate statistics
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    z_score = 1.96  # For 95% CI
    
    lower_bound = mean_predictions - z_score * std_predictions
    upper_bound = mean_predictions + z_score * std_predictions
    
    return mean_predictions, lower_bound, upper_bound


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler: Any,
    n_features: int = 1
) -> np.ndarray:
    """
    Inverse transform predictions to original scale.
    
    Args:
        predictions: Normalized predictions
        scaler: Fitted scaler
        n_features: Number of features
        
    Returns:
        Predictions in original scale
    """
    # Reshape for scaler
    predictions = predictions.reshape(-1, n_features)
    
    # Inverse transform
    original_scale = scaler.inverse_transform(predictions)
    
    return original_scale.flatten()


def get_forecast_dates(
    last_date: datetime,
    horizon: int
) -> List[datetime]:
    """
    Generate dates for forecast period.
    
    Skips weekends (stock market closed).
    
    Args:
        last_date: Last date in historical data
        horizon: Number of trading days to forecast
        
    Returns:
        List of forecast dates
    """
    dates = []
    current_date = last_date
    
    while len(dates) < horizon:
        current_date += timedelta(days=1)
        # Skip weekends
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            dates.append(current_date)
    
    return dates


def plot_predictions(
    forecast_result: ForecastResult,
    title: str = "NVIDIA Stock Price Forecast",
    save_path: Optional[Path] = None,
    show_confidence: bool = True,
    lookback_days: int = 90
) -> plt.Figure:
    """
    Plot historical data and forecast.
    
    Args:
        forecast_result: ForecastResult object
        title: Plot title
        save_path: Optional path to save the plot
        show_confidence: Whether to show confidence intervals
        lookback_days: Number of historical days to show
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=settings.prediction.figsize)
    
    # Historical data (last N days)
    hist_prices = forecast_result.historical_prices[-lookback_days:]
    hist_dates = forecast_result.historical_dates[-lookback_days:]
    
    # Plot historical
    ax.plot(hist_dates, hist_prices, 'b-', linewidth=2, label='Historical')
    
    # Plot forecast
    ax.plot(forecast_result.dates, forecast_result.predictions, 
            'r-', linewidth=2, label='Forecast')
    
    # Mark transition point
    ax.axvline(x=hist_dates[-1], color='gray', linestyle='--', alpha=0.7)
    
    # Confidence interval
    if show_confidence and forecast_result.lower_bound is not None:
        ax.fill_between(
            forecast_result.dates,
            forecast_result.lower_bound,
            forecast_result.upper_bound,
            alpha=0.2,
            color='red',
            label='95% Confidence Interval'
        )
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Forecast plot saved to {save_path}")
    
    return fig


def run_prediction_pipeline(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    horizon: int = 30,
    with_uncertainty: bool = True,
    save_results: bool = True
) -> ForecastResult:
    """
    Run the complete prediction pipeline.
    
    Args:
        run_id: MLflow run ID
        checkpoint_path: Path to model checkpoint
        scaler_path: Path to scaler file
        horizon: Number of days to forecast
        with_uncertainty: Whether to compute uncertainty
        save_results: Whether to save results to files
        
    Returns:
        ForecastResult object
    """
    logger.info("=" * 60)
    logger.info("Starting Prediction Pipeline")
    logger.info("=" * 60)
    
    # Load model
    model, metadata = load_best_model(
        run_id=run_id,
        checkpoint_path=checkpoint_path
    )
    
    # Get sequence length from model or metadata
    sequence_length = metadata.get('params', {}).get('sequence_length', 60)
    if isinstance(sequence_length, str):
        sequence_length = int(sequence_length)
    
    # Load scaler
    scaler = load_scaler(run_id=run_id, scaler_path=scaler_path)
    
    # Load historical data
    df = load_data_from_db(start_year=settings.data.start_year)
    
    # Get last sequence
    last_sequence = get_last_sequence(
        df,
        scaler,
        sequence_length,
        feature_columns=[settings.data.target_column]
    )
    last_sequence = torch.FloatTensor(last_sequence)
    
    # Generate forecast
    device = next(model.parameters()).device
    
    if with_uncertainty:
        mean_preds, lower, upper = generate_forecast_with_uncertainty(
            model, last_sequence, horizon,
            n_samples=settings.prediction.n_samples,
            confidence_level=settings.prediction.confidence_level,
            device=device
        )
        
        # Inverse transform
        predictions = inverse_transform_predictions(mean_preds, scaler)
        lower_bound = inverse_transform_predictions(lower, scaler)
        upper_bound = inverse_transform_predictions(upper, scaler)
    else:
        preds = generate_forecast(model, last_sequence, horizon, device)
        predictions = inverse_transform_predictions(preds, scaler)
        lower_bound = None
        upper_bound = None
    
    # Get dates
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    forecast_dates = get_forecast_dates(last_date, horizon)
    
    # Create result
    result = ForecastResult(
        predictions=predictions,
        dates=forecast_dates,
        historical_prices=df[settings.data.target_column].values,
        historical_dates=pd.to_datetime(df['Date']).tolist(),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        model_run_id=run_id or metadata.get('checkpoint_path'),
        forecast_horizon=horizon
    )
    
    # Save results
    if save_results:
        output_dir = settings.prediction.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save forecast
        forecast_path = output_dir / f'forecast_{timestamp}.{settings.prediction.save_format}'
        result.save(forecast_path, format=settings.prediction.save_format)
        
        # Save plot
        if settings.prediction.plot_predictions:
            plot_path = output_dir / f'forecast_plot_{timestamp}.{settings.prediction.plot_format}'
            fig = plot_predictions(result, save_path=plot_path)
            plt.close(fig)
        
        # Log to MLflow if active
        if mlflow.active_run():
            mlflow.log_artifact(str(forecast_path))
            if settings.prediction.plot_predictions:
                mlflow.log_artifact(str(plot_path))
    
    logger.info("=" * 60)
    logger.info("Prediction Pipeline Complete!")
    logger.info(f"Forecast: {horizon} days ahead")
    logger.info(f"Date range: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    logger.info("=" * 60)
    
    return result


def main():
    """Main entry point for prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate NVIDIA stock price forecast')
    parser.add_argument('--run-id', type=str, default=None, help='MLflow run ID')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--scaler', type=str, default=None, help='Path to scaler file')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon in days')
    parser.add_argument('--no-uncertainty', action='store_true', help='Disable uncertainty estimation')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format
    )
    
    # Run prediction
    result = run_prediction_pipeline(
        run_id=args.run_id,
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        horizon=args.horizon,
        with_uncertainty=not args.no_uncertainty,
        save_results=not args.no_save
    )
    
    # Print forecast
    print("\nForecast Results:")
    print("-" * 50)
    df = result.to_dataframe()
    print(df.to_string(index=False))
    
    print(f"\nForecast Summary:")
    print(f"  First day: {df['Date'].iloc[0].strftime('%Y-%m-%d')} - ${df['Predicted_Price'].iloc[0]:.2f}")
    print(f"  Last day:  {df['Date'].iloc[-1].strftime('%Y-%m-%d')} - ${df['Predicted_Price'].iloc[-1]:.2f}")
    print(f"  Mean:      ${df['Predicted_Price'].mean():.2f}")
    print(f"  Trend:     {'+' if df['Predicted_Price'].iloc[-1] > df['Predicted_Price'].iloc[0] else '-'}"
          f"{abs(df['Predicted_Price'].iloc[-1] - df['Predicted_Price'].iloc[0]):.2f}")


if __name__ == '__main__':
    main()
