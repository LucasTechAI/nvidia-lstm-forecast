"""
Data Preprocessing Module.

This module provides utilities for:
- Loading stock data from SQLite database
- Normalizing features using various scalers
- Creating sequences for LSTM training
- Splitting data into train/validation/test sets
"""

import sqlite3
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import settings, DataConfig

# Configure logging
logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock price sequences.
    
    This dataset wraps sequences of stock data for use with PyTorch DataLoaders.
    """
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            sequences: Input sequences of shape (n_samples, sequence_length, n_features)
            targets: Target values of shape (n_samples, 1) or (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
        if self.targets.dim() == 1:
            self.targets = self.targets.unsqueeze(1)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence, target) tensors
        """
        return self.sequences[idx], self.targets[idx]


def load_data_from_db(
    db_path: Optional[str] = None,
    start_year: int = 2017,
    table_name: str = "nvidia_stock"
) -> pd.DataFrame:
    """
    Load stock data from SQLite database.
    
    Args:
        db_path: Path to SQLite database file. Defaults to config value.
        start_year: Only include data from this year onwards.
        table_name: Name of the table containing stock data.
        
    Returns:
        DataFrame with stock data, sorted by date.
        
    Raises:
        FileNotFoundError: If database file doesn't exist.
        ValueError: If no data found after filtering.
    """
    if db_path is None:
        db_path = settings.database_path
    
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    logger.info(f"Loading data from {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    
    try:
        # Load all data
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        
        logger.info(f"Loaded {len(df)} total records from database")
        
        # Parse date column
        date_column = None
        for col in ['Date', 'date', 'DATE', 'Datetime', 'datetime']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column is None:
            raise ValueError("No date column found in database")
        
        # Parse dates with error handling for mixed timezones
        try:
            df[date_column] = pd.to_datetime(df[date_column], utc=True).dt.tz_localize(None)
        except Exception:
            # Fallback: try parsing without timezone
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            # Remove any rows with invalid dates
            df = df.dropna(subset=[date_column])
        
        df = df.rename(columns={date_column: 'Date'})
        
        # Filter by start year
        df = df[df['Date'].dt.year >= start_year]
        
        if len(df) == 0:
            raise ValueError(f"No data found after filtering for year >= {start_year}")
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Filtered to {len(df)} records from {start_year} onwards")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Log statistics
        _log_data_statistics(df)
        
        return df
        
    finally:
        conn.close()


def _log_data_statistics(df: pd.DataFrame) -> None:
    """Log statistics about the loaded data."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    logger.info("Data Statistics:")
    for col in numeric_cols:
        logger.info(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                   f"mean={df[col].mean():.2f}, std={df[col].std():.2f}")


def normalize_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    scaler_type: str = "MinMaxScaler",
    fit_scaler: bool = True,
    scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
) -> Tuple[np.ndarray, Union[MinMaxScaler, StandardScaler]]:
    """
    Normalize features using specified scaler.
    
    Args:
        df: DataFrame with stock data.
        feature_columns: Columns to normalize. Defaults to ['Close'].
        scaler_type: Type of scaler ("MinMaxScaler" or "StandardScaler").
        fit_scaler: Whether to fit the scaler on data.
        scaler: Pre-fitted scaler to use (if fit_scaler=False).
        
    Returns:
        Tuple of (normalized_data, scaler)
        
    Raises:
        ValueError: If invalid scaler type or missing scaler.
    """
    if feature_columns is None:
        feature_columns = [settings.data.target_column]
    
    # Extract features
    data = df[feature_columns].values.astype(np.float32)
    
    logger.info(f"Normalizing {len(feature_columns)} features: {feature_columns}")
    
    if fit_scaler:
        # Create and fit scaler
        if scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == "StandardScaler":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        normalized_data = scaler.fit_transform(data)
        logger.info(f"Fitted {scaler_type} on {len(data)} samples")
    else:
        if scaler is None:
            raise ValueError("Must provide scaler when fit_scaler=False")
        normalized_data = scaler.transform(data)
        logger.info(f"Transformed data using pre-fitted scaler")
    
    return normalized_data, scaler


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    target_column_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Given normalized data, create overlapping sequences of length `sequence_length`
    where the target is the next value after each sequence.
    
    Args:
        data: Normalized data of shape (n_samples, n_features)
        sequence_length: Number of time steps in each sequence
        target_column_idx: Index of target column in data
        
    Returns:
        Tuple of (X, y) where:
            X: Sequences of shape (n_sequences, sequence_length, n_features)
            y: Targets of shape (n_sequences, 1)
            
    Raises:
        ValueError: If data is too short for sequence length.
    """
    if len(data) <= sequence_length:
        raise ValueError(
            f"Data length ({len(data)}) must be greater than sequence length ({sequence_length})"
        )
    
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length, target_column_idx]
        sequences.append(seq)
        targets.append(target)
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32).reshape(-1, 1)
    
    logger.info(f"Created {len(X)} sequences of length {sequence_length}")
    logger.info(f"Sequence shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split sequences into train/validation/test sets.
    
    Maintains temporal ordering (no shuffling) to avoid data leakage.
    
    Args:
        X: Input sequences
        y: Target values
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (X, y) tuple.
        
    Raises:
        AssertionError: If ratios don't sum to 1.0
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    splits = {
        'train': (X[:train_end], y[:train_end]),
        'val': (X[train_end:val_end], y[train_end:val_end]),
        'test': (X[val_end:], y[val_end:])
    }
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(splits['train'][0])} samples ({train_ratio*100:.0f}%)")
    logger.info(f"  Val:   {len(splits['val'][0])} samples ({val_ratio*100:.0f}%)")
    logger.info(f"  Test:  {len(splits['test'][0])} samples ({test_ratio*100:.0f}%)")
    
    return splits


def create_data_loaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from data splits.
    
    Args:
        splits: Dictionary with train/val/test splits
        batch_size: Batch size for DataLoaders
        shuffle_train: Whether to shuffle training data (each epoch)
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    
    # Only use pin_memory if CUDA is available
    use_pin_memory = torch.cuda.is_available()
    
    for split_name, (X, y) in splits.items():
        dataset = StockDataset(X, y)
        shuffle = shuffle_train if split_name == 'train' else False
        
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Single-threaded for reproducibility
            pin_memory=use_pin_memory
        )
        
        logger.info(f"Created {split_name} DataLoader: {len(loaders[split_name])} batches")
    
    return loaders


def prepare_data_pipeline(
    db_path: Optional[str] = None,
    data_config: Optional[DataConfig] = None,
    sequence_length: int = 60,
    batch_size: int = 32
) -> Tuple[Dict[str, DataLoader], Any, pd.DataFrame]:
    """
    Complete data preparation pipeline.
    
    This function orchestrates the entire data preparation process:
    1. Load data from database
    2. Normalize features
    3. Create sequences
    4. Split into train/val/test
    5. Create DataLoaders
    
    Args:
        db_path: Path to SQLite database
        data_config: Data configuration object
        sequence_length: Length of input sequences
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (dataloaders_dict, scaler, original_dataframe)
    """
    if data_config is None:
        data_config = settings.data
    
    logger.info("=" * 60)
    logger.info("Starting data preparation pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load data
    df = load_data_from_db(
        db_path=db_path,
        start_year=data_config.start_year
    )
    
    # Step 2: Normalize
    feature_columns = data_config.feature_columns or [data_config.target_column]
    normalized_data, scaler = normalize_features(
        df,
        feature_columns=feature_columns,
        scaler_type=data_config.scaler_type
    )
    
    # Step 3: Create sequences
    X, y = create_sequences(
        normalized_data,
        sequence_length=sequence_length
    )
    
    # Step 4: Split data
    splits = train_val_test_split(
        X, y,
        train_ratio=data_config.train_split,
        val_ratio=data_config.val_split,
        test_ratio=data_config.test_split
    )
    
    # Step 5: Create DataLoaders
    loaders = create_data_loaders(splits, batch_size=batch_size)
    
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)
    
    return loaders, scaler, df


def get_last_sequence(
    df: pd.DataFrame,
    scaler: Any,
    sequence_length: int,
    feature_columns: Optional[List[str]] = None
) -> np.ndarray:
    """
    Get the last sequence from data for prediction.
    
    Args:
        df: DataFrame with stock data
        scaler: Fitted scaler for normalization
        sequence_length: Length of sequence to extract
        feature_columns: Columns to include
        
    Returns:
        Last sequence of shape (1, sequence_length, n_features)
    """
    if feature_columns is None:
        feature_columns = [settings.data.target_column]
    
    # Get last sequence_length rows
    last_data = df[feature_columns].tail(sequence_length).values.astype(np.float32)
    
    # Normalize
    normalized = scaler.transform(last_data)
    
    # Reshape for model input
    sequence = normalized.reshape(1, sequence_length, -1)
    
    return sequence


def inverse_transform(
    data: np.ndarray,
    scaler: Any,
    n_features: int = 1
) -> np.ndarray:
    """
    Inverse transform normalized data back to original scale.
    
    Args:
        data: Normalized data to transform
        scaler: Fitted scaler
        n_features: Number of features (for reshaping)
        
    Returns:
        Data in original scale
    """
    # Handle different input shapes
    original_shape = data.shape
    
    if len(original_shape) == 1:
        data = data.reshape(-1, n_features)
    elif len(original_shape) == 3:
        # Flatten (batch, seq, features) to (batch*seq, features)
        data = data.reshape(-1, original_shape[-1])
    
    # Inverse transform
    inversed = scaler.inverse_transform(data)
    
    # Reshape back
    if len(original_shape) == 1:
        return inversed.flatten()
    elif len(original_shape) == 3:
        return inversed.reshape(original_shape)
    
    return inversed
