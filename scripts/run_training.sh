#!/bin/bash
# Run model training with default parameters

set -e

echo "Starting LSTM model training..."

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Run training script
python3 -c "
import logging
import torch
import mlflow
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
from src.models.lstm_model import create_model
from src.training.train import train_model, plot_training_history

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

# Create model
input_size = X.shape[2]
output_size = y.shape[1]

model = create_model(
    input_size=input_size,
    hidden_size=settings.hidden_size,
    num_layers=settings.num_layers,
    dropout=settings.dropout,
    bidirectional=settings.bidirectional,
    output_size=output_size
)
model = model.to(device)

# Setup MLflow
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)

# Training config
config = {
    'batch_size': settings.batch_size,
    'learning_rate': settings.learning_rate,
    'epochs': settings.epochs,
    'early_stopping_patience': settings.early_stopping_patience,
    'optimizer': settings.optimizer
}

# Start MLflow run
with mlflow.start_run(run_name='lstm_training'):
    # Log parameters
    mlflow.log_params({
        'input_size': input_size,
        'hidden_size': settings.hidden_size,
        'num_layers': settings.num_layers,
        'dropout': settings.dropout,
        'bidirectional': settings.bidirectional,
        'sequence_length': settings.sequence_length,
        'batch_size': settings.batch_size,
        'learning_rate': settings.learning_rate,
        'epochs': settings.epochs,
        'optimizer': settings.optimizer
    })
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        config=config,
        device=device,
        mlflow_tracking=True
    )
    
    # Save model
    model_path = settings.model_dir / 'best_model.pth'
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f'Saved model to {model_path}')
    
    # Log model to MLflow
    mlflow.pytorch.log_model(trained_model, 'model')
    
    # Save and log training plots
    plot_path = settings.output_dir / 'training_history.png'
    plot_training_history(
        history['train_loss'],
        history['val_loss'],
        save_path=str(plot_path)
    )
    mlflow.log_artifact(str(plot_path))
    
    # Log scaler
    mlflow.log_artifact(str(scaler_path))
    
    logger.info('Training completed successfully!')
    logger.info(f'Run ID: {mlflow.active_run().info.run_id}')

"

echo "Training completed!"
