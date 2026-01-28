"""Tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.data.preprocessing import (
    normalize_features,
    create_sequences,
    train_val_test_split
)


class TestNormalizeFeatures:
    """Test cases for feature normalization."""
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        normalized, scaler = normalize_features(df, ['A', 'B'])
        
        assert normalized.shape == (5, 2)
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_normalize_range(self):
        """Test normalization range."""
        df = pd.DataFrame({
            'Feature': [0, 50, 100]
        })
        
        normalized, scaler = normalize_features(df, ['Feature'])
        
        assert normalized[0, 0] == 0.0
        assert normalized[2, 0] == 1.0
        assert 0 <= normalized[1, 0] <= 1
    
    def test_normalize_saves_scaler(self):
        """Test that scaler is saved to file."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            scaler_path = f.name
        
        try:
            normalized, scaler = normalize_features(df, ['A'], scaler_path=scaler_path)
            assert Path(scaler_path).exists()
        finally:
            Path(scaler_path).unlink(missing_ok=True)


class TestCreateSequences:
    """Test cases for sequence creation."""
    
    def test_create_sequences_shape(self):
        """Test sequence creation output shape."""
        data = np.random.randn(100, 3)
        sequence_length = 10
        
        X, y = create_sequences(data, sequence_length)
        
        # 100 - 10 - 1 + 1 = 90 sequences
        assert X.shape == (90, 10, 3)
        assert y.shape == (90, 3)
    
    def test_create_sequences_values(self):
        """Test sequence creation maintains data order."""
        data = np.arange(20).reshape(20, 1)
        sequence_length = 5
        
        X, y = create_sequences(data, sequence_length)
        
        # First sequence should be [0,1,2,3,4], target [5]
        assert np.array_equal(X[0], [[0], [1], [2], [3], [4]])
        assert y[0, 0] == 5
    
    def test_create_sequences_with_horizon(self):
        """Test sequence creation with forecast horizon."""
        data = np.arange(20).reshape(20, 1)
        sequence_length = 5
        horizon = 3
        
        X, y = create_sequences(data, sequence_length, forecast_horizon=horizon)
        
        # First sequence [0,1,2,3,4], target should be index 5+3-1=7
        assert X[0][-1, 0] == 4
        assert y[0, 0] == 7


class TestTrainValTestSplit:
    """Test cases for data splitting."""
    
    def test_split_sizes(self):
        """Test split produces correct sizes."""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100, 5)
        
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        assert len(X_train) == 70
        assert len(X_val) == 15
        assert len(X_test) == 15
    
    def test_split_maintains_order(self):
        """Test split maintains temporal order."""
        X = np.arange(100).reshape(100, 1, 1)
        y = np.arange(100).reshape(100, 1)
        
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        # Training should be first 70
        assert X_train[0, 0, 0] == 0
        assert X_train[-1, 0, 0] == 69
        
        # Validation should be next 15
        assert X_val[0, 0, 0] == 70
        
        # Test should be last 15
        assert X_test[0, 0, 0] == 85
    
    def test_split_invalid_ratios(self):
        """Test split raises error for invalid ratios."""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100, 5)
        
        with pytest.raises(ValueError):
            train_val_test_split(X, y, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
