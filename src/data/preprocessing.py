"""
Data preprocessing module for NVIDIA stock data.

This module handles data loading from SQLite, normalization, 
sequence creation, and train/val/test splitting.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

logger = logging.getLogger(__name__)


def load_data_from_db(
    db_path: str, 
    start_year: int = 2017,
    target_column: str = "Close"
) -> pd.DataFrame:
    """
    Load NVIDIA stock data from SQLite database.
    
    Args:
        db_path: Path to SQLite database file
        start_year: Starting year for data filtering
        target_column: Target column name for prediction
        
    Returns:
        DataFrame with filtered stock data
        
    Raises:
        FileNotFoundError: If database file doesn't exist
        ValueError: If no data found after filtering
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    logger.info(f"Loading data from {db_path} (start_year={start_year})")
    
    # Connect to database and load data
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM nvidia_stock"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])
    
    # Filter by year
    if start_year:
        df = df[df['Date'].dt.year >= start_year]
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Validate data
    if len(df) == 0:
        raise ValueError(f"No data found for year >= {start_year}")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    logger.info(f"Loaded {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Available columns: {df.columns.tolist()}")
    logger.info(f"Data statistics:\n{df.describe()}")
    
    return df


def normalize_features(
    df: pd.DataFrame,
    feature_columns: list,
    scaler_type: str = 'MinMaxScaler',
    scaler_path: Optional[str] = None
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize features using specified scaler.
    
    Args:
        df: DataFrame with features
        feature_columns: List of column names to normalize
        scaler_type: Type of scaler ('MinMaxScaler' supported)
        scaler_path: Optional path to save fitted scaler
        
    Returns:
        Tuple of (normalized_data, fitted_scaler)
        
    Raises:
        ValueError: If scaler_type is not supported
    """
    if scaler_type != 'MinMaxScaler':
        raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    logger.info(f"Normalizing features: {feature_columns}")
    
    # Extract features
    data = df[feature_columns].values
    
    # Fit scaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    # Save scaler if path provided
    if scaler_path:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
    
    logger.info(f"Normalized data shape: {normalized_data.shape}")
    logger.info(f"Normalized data range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
    
    return normalized_data, scaler


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Normalized data array (n_samples, n_features)
        sequence_length: Number of time steps in each sequence
        forecast_horizon: Number of steps to predict (default: 1)
        
    Returns:
        Tuple of (X, y) where:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Target values (n_sequences, n_features)
    """
    logger.info(f"Creating sequences (length={sequence_length}, horizon={forecast_horizon})")
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        # Input sequence
        X.append(data[i:i + sequence_length])
        # Target value (next time step after sequence)
        y.append(data[i + sequence_length + forecast_horizon - 1])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created {len(X)} sequences")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into train, validation, and test sets (maintaining temporal order).
    
    Args:
        X: Input sequences
        y: Target values
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    logger.info(f"Split data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_scaler(scaler_path: str) -> MinMaxScaler:
    """
    Load a saved scaler from disk.
    
    Args:
        scaler_path: Path to pickled scaler file
        
    Returns:
        Loaded MinMaxScaler object
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Loaded scaler from {scaler_path}")
    return scaler


def inverse_transform(
    data: np.ndarray,
    scaler: MinMaxScaler
) -> np.ndarray:
    """
    Inverse transform normalized data back to original scale.
    
    Args:
        data: Normalized data
        scaler: Fitted scaler object
        
    Returns:
        Data in original scale
    """
    return scaler.inverse_transform(data)
