"""
Tests for the LSTM model module.
"""

import pytest
import torch
import numpy as np

from src.models.lstm_model import NvidiaLSTM, create_model, count_parameters


class TestNvidiaLSTM:
    """Test cases for NvidiaLSTM model."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly with default parameters."""
        model = NvidiaLSTM()
        
        assert model.input_size == 1
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.output_size == 1
        assert model.dropout_prob == 0.2
        assert model.bidirectional == False
    
    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = NvidiaLSTM(
            input_size=3,
            hidden_size=64,
            num_layers=3,
            output_size=1,
            dropout=0.3,
            bidirectional=True
        )
        
        assert model.input_size == 3
        assert model.hidden_size == 64
        assert model.num_layers == 3
        assert model.dropout_prob == 0.3
        assert model.bidirectional == True
        assert model.num_directions == 2
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 16
        sequence_length = 60
        input_size = 1
        
        model = NvidiaLSTM(input_size=input_size, hidden_size=64, num_layers=2)
        x = torch.randn(batch_size, sequence_length, input_size)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1)
    
    def test_forward_pass_bidirectional(self):
        """Test forward pass with bidirectional LSTM."""
        batch_size = 8
        sequence_length = 30
        input_size = 2
        
        model = NvidiaLSTM(
            input_size=input_size,
            hidden_size=32,
            num_layers=2,
            bidirectional=True
        )
        x = torch.randn(batch_size, sequence_length, input_size)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1)
    
    def test_predict_sequence(self):
        """Test multi-step prediction."""
        model = NvidiaLSTM(input_size=1, hidden_size=32, num_layers=1)
        model.eval()
        
        initial_sequence = torch.randn(1, 60, 1)
        horizon = 10
        
        predictions = model.predict_sequence(initial_sequence, horizon)
        
        assert predictions.shape == (horizon,)
    
    def test_get_config(self):
        """Test that get_config returns correct configuration."""
        model = NvidiaLSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False
        )
        
        config = model.get_config()
        
        assert config['input_size'] == 1
        assert config['hidden_size'] == 128
        assert config['num_layers'] == 2
        assert config['dropout'] == 0.2
        assert config['bidirectional'] == False
        assert 'num_parameters' in config
        assert config['num_parameters'] > 0
    
    def test_model_on_device(self):
        """Test model can be moved to device."""
        model = NvidiaLSTM(hidden_size=32, num_layers=1)
        model = model.to('cpu')
        
        assert next(model.parameters()).device.type == 'cpu'
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the model."""
        model = NvidiaLSTM(input_size=1, hidden_size=32, num_layers=1)
        
        x = torch.randn(4, 30, 1, requires_grad=True)
        y = torch.randn(4, 1)
        
        output = model(x)
        loss = torch.nn.MSELoss()(output, y)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestCreateModel:
    """Test cases for create_model factory function."""
    
    def test_create_model_default(self):
        """Test create_model with default config."""
        model = create_model(device='cpu')
        
        assert isinstance(model, NvidiaLSTM)
        assert next(model.parameters()).device.type == 'cpu'


class TestCountParameters:
    """Test cases for count_parameters utility."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = NvidiaLSTM(input_size=1, hidden_size=32, num_layers=1)
        
        counts = count_parameters(model)
        
        assert 'total' in counts
        assert 'trainable' in counts
        assert 'non_trainable' in counts
        assert counts['total'] > 0
        assert counts['trainable'] == counts['total']  # All params trainable by default
        assert counts['non_trainable'] == 0
