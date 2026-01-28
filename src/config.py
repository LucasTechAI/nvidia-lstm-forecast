"""
Centralized Configuration Module.

This module provides a single source of truth for all configuration
values, paths, and settings used throughout the application.
"""

from dotenv import load_dotenv
from pathlib import Path
import os

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Path Configuration
# ============================================================================

# Project root directory (where pyproject.toml lives)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Key directories
DATA_DIR = ROOT_DIR / "data"

# ============================================================================
# Database Configuration
# ============================================================================

DATABASE_PATH = os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "nvidia.db"))


# ============================================================================
# Stock Data Configuration
# ============================================================================

STOCK_SYMBOL = "NVDA"
DEFAULT_DATA_PERIOD = "2y"
DEFAULT_DATA_INTERVAL = "1d"


# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================================
# Model Configuration
# ============================================================================

# Data Parameters
DATA_START_YEAR = 2017
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
TARGET_COLUMN = "Close"

# LSTM Architecture
SEQUENCE_LENGTH = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = False

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
OPTIMIZER = "Adam"
LOSS_FUNCTION = "MSE"
EARLY_STOPPING_PATIENCE = 10

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(ROOT_DIR / "mlruns"))
MLFLOW_EXPERIMENT_NAME = "nvidia-lstm-forecast"
MLFLOW_ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION", None)

# Prediction
FORECAST_HORIZON = 30

# Model and Output Directories
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
MLRUNS_DIR = ROOT_DIR / "mlruns"


# ============================================================================
# Settings Class (Pydantic-style for validation)
# ============================================================================

class Settings:
    """
    Application settings container.
    
    This class provides a structured way to access all configuration values
    with type hints and documentation.
    """
    
    # Paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    model_dir: Path = MODEL_DIR
    output_dir: Path = OUTPUT_DIR
    mlruns_dir: Path = MLRUNS_DIR
    
    # Database
    database_path: str = DATABASE_PATH
    
    # Stock
    stock_symbol: str = STOCK_SYMBOL
    default_period: str = DEFAULT_DATA_PERIOD
    default_interval: str = DEFAULT_DATA_INTERVAL
    
    # Logging
    log_level: str = LOG_LEVEL
    log_format: str = LOG_FORMAT
    
    # Data Parameters
    data_start_year: int = DATA_START_YEAR
    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    test_split: float = TEST_SPLIT
    target_column: str = TARGET_COLUMN
    
    # LSTM Architecture
    sequence_length: int = SEQUENCE_LENGTH
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    bidirectional: bool = BIDIRECTIONAL
    
    # Training
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    optimizer: str = OPTIMIZER
    loss_function: str = LOSS_FUNCTION
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    
    # MLflow
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
    mlflow_experiment_name: str = MLFLOW_EXPERIMENT_NAME
    mlflow_artifact_location: str = MLFLOW_ARTIFACT_LOCATION
    
    # Prediction
    forecast_horizon: int = FORECAST_HORIZON
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings instance from environment variables."""
        return cls()


# Singleton instance
settings = Settings()
