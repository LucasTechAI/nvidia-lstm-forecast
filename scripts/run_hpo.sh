#!/bin/bash
# Run hyperparameter optimization

set -e

echo "Starting hyperparameter optimization..."

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Number of trials (can be overridden)
N_TRIALS=${1:-20}

echo "Running $N_TRIALS optimization trials..."

# Run HPO script
python3 -c "
import logging
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, '${PROJECT_ROOT}')

from src.config import settings
from src.data.preprocessing import (
    load_data_from_db,
    normalize_features,
    create_sequences,
    train_val_test_split
)
from src.training.hyperparameter_search import run_hyperparameter_search, save_study

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Create directories
Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
Path(settings.mlruns_dir).mkdir(parents=True, exist_ok=True)

# Load data
logger.info('Loading data...')
df = load_data_from_db(
    settings.database_path,
    start_year=settings.data_start_year,
    target_column=settings.target_column
)

# Prepare features
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
available_features = [col for col in feature_columns if col in df.columns]
logger.info(f'Using features: {available_features}')

# Normalize
scaler_path = settings.model_dir / 'scaler.pkl'
normalized_data, scaler = normalize_features(
    df,
    available_features,
    scaler_path=str(scaler_path)
)

# Create sequences
X, y = create_sequences(
    normalized_data,
    sequence_length=settings.sequence_length,
    forecast_horizon=1
)

# Split data
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
    X, y,
    train_ratio=settings.train_split,
    val_ratio=settings.val_split,
    test_ratio=settings.test_split
)

input_size = X.shape[2]
output_size = y.shape[1]

# Run hyperparameter search
logger.info('Starting hyperparameter optimization...')
study, best_params = run_hyperparameter_search(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    input_size=input_size,
    output_size=output_size,
    n_trials=$N_TRIALS,
    device=device,
    mlflow_tracking_uri=settings.mlflow_tracking_uri,
    experiment_name=settings.mlflow_experiment_name,
    study_name='nvidia_lstm_hpo'
)

# Save study
study_path = settings.output_dir / 'optuna_study.pkl'
save_study(study, str(study_path))

logger.info('Hyperparameter optimization completed!')
logger.info(f'Best parameters: {best_params}')

"

echo "Hyperparameter optimization completed!"
