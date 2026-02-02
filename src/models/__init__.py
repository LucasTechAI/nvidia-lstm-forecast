"""
Models module for NVIDIA LSTM Forecast.

This module contains the LSTM model architecture for stock price prediction.
"""

from .lstm_model import NvidiaLSTM, create_model

__all__ = ["NvidiaLSTM", "create_model"]
