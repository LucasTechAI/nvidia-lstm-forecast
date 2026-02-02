"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    np.random.seed(42)
    n_samples = 500
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate realistic-looking stock data
    base_price = 200
    returns = np.random.randn(n_samples) * 0.02
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(n_samples) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_samples),
        'Adj Close': prices
    })
    
    return df


@pytest.fixture
def sample_sequences():
    """Create sample sequences for testing."""
    np.random.seed(42)
    
    n_samples = 100
    sequence_length = 60
    n_features = 1
    
    X = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    y = np.random.randn(n_samples, 1).astype(np.float32)
    
    return X, y


@pytest.fixture
def sample_model():
    """Create a sample LSTM model for testing."""
    from src.models.lstm_model import NvidiaLSTM
    
    model = NvidiaLSTM(
        input_size=1,
        hidden_size=32,
        num_layers=1,
        dropout=0.0
    )
    
    return model


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cpu')
