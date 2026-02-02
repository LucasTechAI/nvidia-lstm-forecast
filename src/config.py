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
# Data Processing Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data filtering
    start_year: int = 2017
    
    # Train/Val/Test split ratios (must sum to 1.0)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Target column for prediction (lowercase to match database)
    target_column: str = "close"
    
    # Feature columns to use (None = use only target)
    feature_columns: Optional[List[str]] = None
    
    # Scaler type for normalization
    scaler_type: str = "MinMaxScaler"  # Options: "MinMaxScaler", "StandardScaler"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Train, val, and test splits must sum to 1.0"


# ============================================================================
# LSTM Model Configuration
# ============================================================================

@dataclass
class LSTMConfig:
    """Configuration for LSTM model architecture."""
    
    # Sequence parameters
    sequence_length: int = 60  # Number of days to look back
    
    # Architecture
    input_size: int = 1  # Number of input features
    hidden_size: int = 128  # LSTM hidden state size
    num_layers: int = 2  # Number of stacked LSTM layers
    dropout: float = 0.2  # Dropout probability between layers
    bidirectional: bool = False  # Use bidirectional LSTM
    
    # Output
    output_size: int = 1  # Number of output features (predict Close price)
    
    @property
    def num_directions(self) -> int:
        """Get number of directions (1 for unidirectional, 2 for bidirectional)."""
        return 2 if self.bidirectional else 1


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    
    # Optimizer
    optimizer: str = "Adam"  # Options: "Adam", "AdamW", "SGD", "RMSprop"
    weight_decay: float = 1e-5  # L2 regularization
    
    # Loss function
    loss_function: str = "MSE"  # Options: "MSE", "MAE", "Huber"
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "ReduceLROnPlateau"  # Options: "ReduceLROnPlateau", "StepLR", "CosineAnnealing"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Gradient clipping
    gradient_clip_value: Optional[float] = 1.0
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: MODELS_DIR / "checkpoints")
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Device
    device: str = "auto"  # Options: "auto", "cuda", "cpu"


# ============================================================================
# MLflow Configuration
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
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    outputs_dir: Path = OUTPUTS_DIR
    
    # Database
    database_path: str = DATABASE_PATH
    
    # Stock
    stock_symbol: str = STOCK_SYMBOL
    default_period: str = DEFAULT_DATA_PERIOD
    default_interval: str = DEFAULT_DATA_INTERVAL
    
    # Logging
    log_level: str = LOG_LEVEL
    log_format: str = LOG_FORMAT
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    
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
