"""
LSTM model for NVIDIA stock price prediction.

This module implements a configurable LSTM architecture using PyTorch.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class NvidiaLSTM(nn.Module):
    """
    LSTM model for stock price prediction.
    
    This model consists of stacked LSTM layers with dropout,
    followed by a fully-connected output layer.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
            output_size: Number of output features
        """
        super(NvidiaLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Dropout only for multi-layer LSTM
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Fully connected output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized NvidiaLSTM: input={input_size}, hidden={hidden_size}, "
                   f"layers={num_layers}, dropout={dropout}, bidirectional={bidirectional}")
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize FC layer
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last time step's output
        # Shape: (batch_size, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layer
        # Shape: (batch_size, output_size)
        output = self.fc(last_output)
        
        return output
    
    def predict_sequence(
        self,
        initial_sequence: torch.Tensor,
        n_steps: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate multi-step predictions iteratively.
        
        Args:
            initial_sequence: Initial input sequence (1, sequence_length, input_size)
            n_steps: Number of future steps to predict
            device: Device to run predictions on
            
        Returns:
            Tensor of predictions (n_steps, input_size)
        """
        self.eval()
        predictions = []
        
        # Start with initial sequence
        current_seq = initial_sequence.clone().to(device)
        
        with torch.no_grad():
            for _ in range(n_steps):
                # Predict next step
                pred = self.forward(current_seq)
                predictions.append(pred.cpu())
                
                # Update sequence: remove first step, add prediction as last step
                # pred shape: (1, output_size), need to expand to (1, 1, output_size)
                pred_expanded = pred.unsqueeze(1)
                
                # Roll the sequence and add new prediction
                current_seq = torch.cat([current_seq[:, 1:, :], pred_expanded], dim=1)
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=0)
        return predictions
    
    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    output_size: int = 1
) -> NvidiaLSTM:
    """
    Factory function to create an LSTM model.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        output_size: Number of output features
        
    Returns:
        Initialized NvidiaLSTM model
    """
    model = NvidiaLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        output_size=output_size
    )
    
    num_params = model.get_num_parameters()
    logger.info(f"Created model with {num_params:,} trainable parameters")
    
    return model
