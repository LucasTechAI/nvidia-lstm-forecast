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
    
    
    # Database
    database_path: str = DATABASE_PATH
    
    # Stock
    stock_symbol: str = STOCK_SYMBOL
    default_period: str = DEFAULT_DATA_PERIOD
    default_interval: str = DEFAULT_DATA_INTERVAL
    
    # Logging
    log_level: str = LOG_LEVEL
    log_format: str = LOG_FORMAT
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings instance from environment variables."""
        return cls()


# Singleton instance
settings = Settings()
