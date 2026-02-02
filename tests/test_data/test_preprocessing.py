"""
Tests for the data preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from etl.preprocessing import (
    StockDataset,
    create_sequences,
    train_val_test_split,
    normalize_features,
)


class TestStockDataset:
    """Test cases for StockDataset class."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        sequences = np.random.randn(100, 60, 1).astype(np.float32)
        targets = np.random.randn(100).astype(np.float32)
        
        dataset = StockDataset(sequences, targets)
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        sequences = np.random.randn(100, 60, 1).astype(np.float32)
        targets = np.random.randn(100).astype(np.float32)
        
        dataset = StockDataset(sequences, targets)
        seq, target = dataset[0]
        
        assert isinstance(seq, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert seq.shape == (60, 1)
        assert target.shape == (1,)
    
    def test_dataset_target_shape_2d(self):
        """Test dataset with 2D targets."""
        sequences = np.random.randn(50, 30, 2).astype(np.float32)
        targets = np.random.randn(50, 1).astype(np.float32)
        
        dataset = StockDataset(sequences, targets)
        _, target = dataset[0]
        
        assert target.shape == (1,)


class TestCreateSequences:
    """Test cases for create_sequences function."""
    
    def test_create_sequences_basic(self):
        """Test basic sequence creation."""
        data = np.random.randn(100, 1).astype(np.float32)
        sequence_length = 10
        
        X, y = create_sequences(data, sequence_length)
        
        assert X.shape == (90, 10, 1)
        assert y.shape == (90, 1)
    
    def test_create_sequences_multivariate(self):
        """Test sequence creation with multiple features."""
        data = np.random.randn(100, 3).astype(np.float32)
        sequence_length = 20
        
        X, y = create_sequences(data, sequence_length)
        
        assert X.shape == (80, 20, 3)
        assert y.shape == (80, 1)
    
    def test_create_sequences_values(self):
        """Test that sequence values are correct."""
        data = np.arange(20).reshape(-1, 1).astype(np.float32)
        sequence_length = 5
        
        X, y = create_sequences(data, sequence_length)
        
        # First sequence should be [0, 1, 2, 3, 4], target = 5
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3, 4])
        assert y[0, 0] == 5
        
        # Second sequence should be [1, 2, 3, 4, 5], target = 6
        np.testing.assert_array_equal(X[1, :, 0], [1, 2, 3, 4, 5])
        assert y[1, 0] == 6
    
    def test_create_sequences_too_short(self):
        """Test that short data raises error."""
        data = np.random.randn(10, 1).astype(np.float32)
        sequence_length = 15
        
        with pytest.raises(ValueError):
            create_sequences(data, sequence_length)


class TestTrainValTestSplit:
    """Test cases for train_val_test_split function."""
    
    def test_split_ratios(self):
        """Test that split ratios are respected."""
        X = np.random.randn(100, 60, 1)
        y = np.random.randn(100, 1)
        
        splits = train_val_test_split(X, y, 0.7, 0.15, 0.15)
        
        assert len(splits['train'][0]) == 70
        assert len(splits['val'][0]) == 15
        assert len(splits['test'][0]) == 15
    
    def test_split_no_shuffle(self):
        """Test that temporal order is preserved (no shuffling)."""
        X = np.arange(100).reshape(100, 1, 1)
        y = np.arange(100).reshape(100, 1)
        
        splits = train_val_test_split(X, y, 0.7, 0.15, 0.15)
        
        # Training should have first 70 samples
        assert splits['train'][0][0, 0, 0] == 0
        assert splits['train'][0][-1, 0, 0] == 69
        
        # Validation should have next 15 samples
        assert splits['val'][0][0, 0, 0] == 70
        
        # Test should have last 15 samples
        assert splits['test'][0][-1, 0, 0] == 99
    
    def test_split_invalid_ratios(self):
        """Test that invalid ratios raise error."""
        X = np.random.randn(100, 60, 1)
        y = np.random.randn(100, 1)
        
        with pytest.raises(AssertionError):
            train_val_test_split(X, y, 0.5, 0.3, 0.3)  # Sum > 1


class TestNormalizeFeatures:
    """Test cases for normalize_features function."""
    
    def test_normalize_minmax(self):
        """Test MinMaxScaler normalization."""
        df = pd.DataFrame({
            'Close': np.random.randn(100) * 100 + 200
        })
        
        normalized, scaler = normalize_features(df, ['Close'], 'MinMaxScaler')
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_normalize_standard(self):
        """Test StandardScaler normalization."""
        df = pd.DataFrame({
            'Close': np.random.randn(100) * 100 + 200
        })
        
        normalized, scaler = normalize_features(df, ['Close'], 'StandardScaler')
        
        # StandardScaler centers around 0 with std ~1
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1) < 0.1
    
    def test_normalize_inverse(self):
        """Test that inverse transform works correctly."""
        df = pd.DataFrame({
            'Close': np.array([100, 200, 300, 400, 500], dtype=np.float32)
        })
        
        normalized, scaler = normalize_features(df, ['Close'], 'MinMaxScaler')
        inversed = scaler.inverse_transform(normalized)
        
        np.testing.assert_array_almost_equal(inversed.flatten(), df['Close'].values, decimal=4)
