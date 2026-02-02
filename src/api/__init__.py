"""
FastAPI Application for NVIDIA Stock Price Prediction.

This module provides REST API endpoints for:
- Model inference and predictions
- Model training
- Data retrieval
- Health checks
"""

from src.api.main import app

__all__ = ["app"]
