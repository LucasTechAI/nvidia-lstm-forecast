"""
LSTM Model for NVIDIA Stock Price Prediction.

This module implements a configurable LSTM neural network for time series
forecasting, with support for:
- Multiple stacked LSTM layers
- Bidirectional processing
- Dropout regularization
- Various weight initialization strategies
"""

import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from src.config import settings, LSTMConfig

# Configure logging
logger = logging.getLogger(__name__)


class NvidiaLSTM(nn.Module):
    """
    LSTM-based neural network for NVIDIA stock price prediction.
    
    Architecture:
        Input -> LSTM (stacked layers) -> Dropout -> Fully Connected -> Output
    
    Features:
        - Configurable number of LSTM layers
        - Optional bidirectional processing
        - Dropout between layers for regularization
        - Xavier/Kaiming weight initialization
    
    Args:
        input_size: Number of input features (default: 1 for Close price)
        hidden_size: Number of hidden units in LSTM layers
        num_layers: Number of stacked LSTM layers
        output_size: Number of output values (default: 1)
        dropout: Dropout probability between layers
        bidirectional: Whether to use bidirectional LSTM
        
    Example:
        >>> model = NvidiaLSTM(input_size=1, hidden_size=128, num_layers=2)
        >>> x = torch.randn(32, 60, 1)  # (batch, seq_len, features)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 1])
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(NvidiaLSTM, self).__init__()
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Log model configuration
        logger.info(f"Initializing NvidiaLSTM:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Bidirectional: {bidirectional}")
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Dropout only between layers
            bidirectional=bidirectional
        )
        
        # Dropout layer (applied after LSTM)
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully connected output layer
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, output_size)
        
        # Initialize weights
        self._init_weights()
        
        # Calculate and log total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def _init_weights(self) -> None:
        """
        Initialize model weights using Xavier/Glorot initialization.
        
        LSTM weights use orthogonal initialization for better gradient flow.
        Linear layers use Xavier initialization.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: Zero, except forget gate (set to 1)
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (improves training stability)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        
        # FC layer: Xavier initialization
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        logger.debug("Weights initialized using Xavier/Orthogonal initialization")
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional initial hidden state (h_0, c_0)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # (h_n, c_n): final hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Use the last time step output
        # For bidirectional, concatenate forward and backward outputs
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            last_output = torch.cat([
                lstm_out[:, -1, :self.hidden_size],  # Forward last step
                lstm_out[:, 0, self.hidden_size:]    # Backward first step (last of reverse)
            ], dim=1)
        else:
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out
    
    def _init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states to zeros.
        
        Args:
            batch_size: Size of the batch
            device: Device to create tensors on
            
        Returns:
            Tuple of (h_0, c_0) initialized to zeros
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        return (h_0, c_0)
    
    def predict_sequence(
        self,
        initial_sequence: torch.Tensor,
        horizon: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate predictions for multiple future time steps.
        
        This method performs iterative prediction where each predicted value
        is appended to the sequence and used to predict the next value.
        
        Args:
            initial_sequence: Starting sequence of shape (1, seq_len, input_size)
            horizon: Number of future time steps to predict
            device: Device to run predictions on
            
        Returns:
            Predictions tensor of shape (horizon,)
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        predictions = []
        
        # Copy the initial sequence
        current_sequence = initial_sequence.clone().to(device)
        
        with torch.no_grad():
            for _ in range(horizon):
                # Predict next value
                pred = self.forward(current_sequence)
                predictions.append(pred.item())
                
                # Update sequence: remove first, append prediction
                # Create new input with prediction
                new_input = pred.view(1, 1, self.input_size)
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    new_input
                ], dim=1)
        
        return torch.tensor(predictions, device=device)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as a dictionary.
        
        Useful for logging to MLflow or saving with checkpoints.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout_prob,
            'bidirectional': self.bidirectional,
            'num_parameters': sum(p.numel() for p in self.parameters()),
        }


def create_model(
    config: Optional[LSTMConfig] = None,
    device: Optional[str] = None
) -> NvidiaLSTM:
    """
    Factory function to create an LSTM model from configuration.
    
    Args:
        config: LSTM configuration object. Uses settings.lstm if None.
        device: Device to move model to. Uses settings.get_device() if None.
        
    Returns:
        Initialized NvidiaLSTM model on specified device
    """
    if config is None:
        config = settings.lstm
    
    if device is None:
        device = settings.get_device()
    
    model = NvidiaLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size,
        dropout=config.dropout,
        bidirectional=config.bidirectional
    )
    
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None
) -> NvidiaLSTM:
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model to
        
    Returns:
        Loaded NvidiaLSTM model
    """
    if device is None:
        device = settings.get_device()
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    
    # Create model with saved configuration
    model = NvidiaLSTM(
        input_size=model_config.get('input_size', 1),
        hidden_size=model_config.get('hidden_size', 128),
        num_layers=model_config.get('num_layers', 2),
        output_size=model_config.get('output_size', 1),
        dropout=model_config.get('dropout', 0.2),
        bidirectional=model_config.get('bidirectional', False)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model loaded successfully to {device}")
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }
