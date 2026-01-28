#!/bin/bash
# Generate predictions with the best model

set -e

echo "Generating predictions..."

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# MLflow run ID (should be provided as argument)
RUN_ID=${1:-""}

if [ -z "$RUN_ID" ]; then
    echo "Usage: ./run_prediction.sh <mlflow_run_id>"
    echo "Please provide the MLflow run ID of the trained model"
    exit 1
fi

echo "Using MLflow run ID: $RUN_ID"

# Run prediction script
python3 -c "
import logging
import torch
import pandas as pd
from pathlib import Path
from datetime import timedelta
import sys

# Add src to path
sys.path.insert(0, '${PROJECT_ROOT}')

from src.config import settings
from src.data.preprocessing import load_data_from_db, normalize_features, load_scaler
from src.prediction.predict import (
    load_best_model,
    generate_forecast,
    inverse_transform_predictions,
    plot_predictions,
    save_predictions_to_csv
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Create output directory
Path(settings.output_dir).mkdir(parents=True, exist_ok=True)

# Load historical data
logger.info('Loading historical data...')
df = load_data_from_db(
    settings.database_path,
    start_year=settings.data_start_year,
    target_column=settings.target_column
)

# Prepare features
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
available_features = [col for col in feature_columns if col in df.columns]

# Load scaler
scaler_path = settings.model_dir / 'scaler.pkl'
scaler = load_scaler(str(scaler_path))

# Normalize last sequence
normalized_data, _ = normalize_features(df, available_features)
last_sequence = normalized_data[-settings.sequence_length:]

# Load model
logger.info('Loading model from MLflow...')
model = load_best_model(
    mlflow_run_id='$RUN_ID',
    mlflow_tracking_uri=settings.mlflow_tracking_uri,
    device=device
)

# Generate forecast
logger.info(f'Generating {settings.forecast_horizon}-day forecast...')
forecast_normalized = generate_forecast(
    model=model,
    last_sequence=last_sequence,
    horizon=settings.forecast_horizon,
    device=device
)

# Inverse transform
forecast = inverse_transform_predictions(
    forecast_normalized,
    str(scaler_path)
)

# Create forecast dates
last_date = df['Date'].iloc[-1]
forecast_dates = pd.date_range(
    start=last_date + timedelta(days=1),
    periods=settings.forecast_horizon,
    freq='D'
)

# Save predictions
csv_path = settings.output_dir / 'forecast.csv'
save_predictions_to_csv(
    forecast,
    forecast_dates,
    str(csv_path),
    column_names=available_features
)

# Plot predictions
plot_path = settings.output_dir / 'forecast_plot.png'

# Extract close price for plotting
close_idx = available_features.index(settings.target_column)
forecast_close = forecast[:, close_idx]

plot_predictions(
    historical_data=df,
    forecast_data=forecast_close,
    forecast_dates=forecast_dates,
    target_column=settings.target_column,
    save_path=str(plot_path),
    show_last_n_days=180
)

logger.info('Prediction completed successfully!')
logger.info(f'Forecast saved to {csv_path}')
logger.info(f'Plot saved to {plot_path}')

# Print forecast summary
print('\nForecast Summary:')
print(f'Horizon: {settings.forecast_horizon} days')
print(f'First prediction: {forecast_close[0]:.2f}')
print(f'Last prediction: {forecast_close[-1]:.2f}')
print(f'Mean prediction: {forecast_close.mean():.2f}')
print(f'Min prediction: {forecast_close.min():.2f}')
print(f'Max prediction: {forecast_close.max():.2f}')

"

echo "Prediction completed!"
