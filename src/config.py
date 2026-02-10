"""
Centralized Configuration Module.

This module provides a single source of truth for all configuration
values, paths, and settings used throughout the application.
"""

from dotenv import load_dotenv
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Path Configuration
# ============================================================================

# Project root directory (where pyproject.toml lives)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Key directories
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
OUTPUTS_DIR = DATA_DIR / "outputs"
MLRUNS_DIR = DATA_DIR / "mlruns"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR, MLRUNS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Database Configuration
# ============================================================================

DATABASE_PATH = os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "nvidia_stock.db"))


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
PREDICTION_UNCERTAINTY_FACTOR = 0.10  # 10% uncertainty for confidence intervals

# Model and Output Directories
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
MLRUNS_DIR = ROOT_DIR / "mlruns"


# ============================================================================
# Settings Class (Pydantic-style for validation)
# ============================================================================

@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""
    
    # Tracking server - use SQLite database backend (filesystem backend deprecated Feb 2026)
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLRUNS_DIR}/mlflow.db")
    
    # Experiment settings
    experiment_name: str = "nvidia-lstm-forecast"
    
    # Artifact storage
    artifact_location: Optional[str] = os.getenv("MLFLOW_ARTIFACT_ROOT", str(MLRUNS_DIR / "artifacts"))
    
    # Run settings
    run_name_prefix: str = "lstm_run"
    
    # Logging settings
    log_models: bool = True
    log_artifacts: bool = True
    log_system_metrics: bool = True
    
    # Model registry
    registered_model_name: str = "nvidia-lstm-model"


# ============================================================================
# Hyperparameter Optimization Configuration
# ============================================================================

@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization with Optuna."""
    
    # Study settings
    study_name: str = "nvidia-lstm-hpo"
    n_trials: int = 50
    timeout: Optional[int] = None  # Timeout in seconds
    
    # Optimization direction
    direction: str = "minimize"  # Minimize validation RMSE
    metric: str = "val_rmse"
    
    # Search space bounds
    num_layers_range: tuple = (1, 4)
    hidden_size_choices: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    learning_rate_range: tuple = (1e-5, 1e-2)  # Log scale
    dropout_range: tuple = (0.1, 0.5)
    sequence_length_choices: List[int] = field(default_factory=lambda: [30, 60, 90, 120])
    batch_size_choices: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    
    # Optuna sampler and pruner
    sampler: str = "TPE"  # Options: "TPE", "CMA-ES", "Random"
    pruner: str = "MedianPruner"  # Options: "MedianPruner", "HyperbandPruner", "None"
    
    # Storage
    storage: Optional[str] = field(default_factory=lambda: f"sqlite:///{MODELS_DIR}/optuna.db")
    
    # Parallelization
    n_jobs: int = 1  # Number of parallel trials


# ============================================================================
# Prediction Configuration
# ============================================================================

@dataclass
class PredictionConfig:
    """Configuration for model inference and forecasting."""
    
    # Forecast horizon
    forecast_horizon: int = 30  # Days to predict ahead
    
    # Confidence intervals (for uncertainty estimation)
    confidence_level: float = 0.95
    n_samples: int = 100  # Monte Carlo samples for uncertainty
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: OUTPUTS_DIR / "predictions")
    save_format: str = "csv"  # Options: "csv", "json", "parquet"
    
    # Visualization
    plot_predictions: bool = True
    plot_format: str = "png"  # Options: "png", "svg", "pdf"
    figsize: tuple = (14, 7)


# ============================================================================
# Settings Class (Aggregated Configuration)
# ============================================================================

@dataclass
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
    prediction_uncertainty_factor: float = PREDICTION_UNCERTAINTY_FACTOR
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings instance from environment variables."""
        return cls()
    
    def get_device(self) -> str:
        """Get the appropriate device (cuda/cpu) based on configuration and availability."""
        import torch
        if self.training.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.training.device


# Singleton instance
settings = Settings()
