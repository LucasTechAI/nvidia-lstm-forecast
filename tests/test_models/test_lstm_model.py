"""Tests for LSTM model module."""

import torch
import torch.nn as nn
import pytest

from src.models.lstm_model import NvidiaLSTM, create_model


class TestNvidiaLSTM:
    """Test cases for NvidiaLSTM model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = NvidiaLSTM(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            output_size=5
        )
        
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_model_forward_pass(self):
        """Test forward pass with correct shapes."""
        model = NvidiaLSTM(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=5
        )
        
        # Batch of 8, sequence length 10, 5 features
        x = torch.randn(8, 10, 5)
        output = model(x)
        
        assert output.shape == (8, 5)
    
    def test_model_different_sizes(self):
        """Test model works with different input sizes."""
        for input_size in [1, 3, 5, 10]:
            for hidden_size in [32, 64, 128]:
                model = NvidiaLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=2,
                    output_size=input_size
                )
                
                x = torch.randn(4, 20, input_size)
                output = model(x)
                
                assert output.shape == (4, input_size)
    
    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM."""
        model = NvidiaLSTM(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            output_size=5
        )
        
        x = torch.randn(8, 10, 5)
        output = model(x)
        
        assert output.shape == (8, 5)
    
    def test_single_layer_lstm(self):
        """Test single layer LSTM (no dropout)."""
        model = NvidiaLSTM(
            input_size=5,
            hidden_size=64,
            num_layers=1,
            dropout=0.2,  # Should be ignored for single layer
            output_size=5
        )
        
        x = torch.randn(8, 10, 5)
        output = model(x)
        
        assert output.shape == (8, 5)
    
    def test_get_num_parameters(self):
        """Test parameter counting."""
        model = NvidiaLSTM(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=5
        )
        
        num_params = model.get_num_parameters()
        
        assert num_params > 0
        assert isinstance(num_params, int)
    
    def test_predict_sequence(self):
        """Test multi-step prediction."""
        model = NvidiaLSTM(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=5
        )
        
        # Initial sequence: (1, sequence_length, input_size)
        initial_seq = torch.randn(1, 10, 5)
        n_steps = 5
        
        predictions = model.predict_sequence(
            initial_seq,
            n_steps=n_steps,
            device=torch.device('cpu')
        )
        
        assert predictions.shape == (n_steps, 5)


class TestCreateModel:
    """Test cases for create_model factory function."""
    
    def test_create_model_default(self):
        """Test creating model with default parameters."""
        model = create_model(input_size=5)
        
        assert model is not None
        assert isinstance(model, NvidiaLSTM)
    
    def test_create_model_custom(self):
        """Test creating model with custom parameters."""
        model = create_model(
            input_size=3,
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            output_size=3
        )
        
        assert model.input_size == 3
        assert model.hidden_size == 256
        assert model.num_layers == 3
        assert model.dropout == 0.3
        assert model.bidirectional == True
        assert model.output_size == 3
