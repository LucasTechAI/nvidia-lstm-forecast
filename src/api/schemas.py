"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# ============== Health Check ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status", examples=["healthy"])
    model_loaded: bool = Field(..., description="Whether model is loaded")
    database_connected: bool = Field(..., description="Whether database is connected")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============== Prediction ==============

class PredictRequest(BaseModel):
    """Request for generating predictions."""
    horizon: int = Field(default=30, ge=1, le=365, description="Number of days to predict")
    with_uncertainty: bool = Field(default=True, description="Include confidence intervals")
    n_samples: int = Field(default=100, ge=10, le=500, description="Monte Carlo samples for uncertainty")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals")


class PredictionItem(BaseModel):
    """Single prediction item."""
    date: datetime
    predicted_price: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class PredictResponse(BaseModel):
    """Response with predictions."""
    predictions: List[PredictionItem]
    last_known_price: float
    last_known_date: datetime
    forecast_horizon: int
    model_info: dict
    generated_at: datetime = Field(default_factory=datetime.now)


# ============== Inference (Single-step) ==============

class InferenceRequest(BaseModel):
    """Request for single-step inference with custom sequence."""
    sequence: List[float] = Field(..., min_length=1, description="Input sequence of prices")
    steps: int = Field(default=1, ge=1, le=30, description="Number of steps to predict")


class InferenceResponse(BaseModel):
    """Response with inference results."""
    predictions: List[float]
    input_length: int
    generated_at: datetime = Field(default_factory=datetime.now)


# ============== Training ==============

class TrainRequest(BaseModel):
    """Request to start training."""
    epochs: Optional[int] = Field(default=None, ge=1, le=500, description="Number of epochs")
    batch_size: Optional[int] = Field(default=None, ge=8, le=256, description="Batch size")
    learning_rate: Optional[float] = Field(default=None, ge=1e-6, le=1e-1, description="Learning rate")
    hidden_size: Optional[int] = Field(default=None, ge=32, le=512, description="LSTM hidden size")
    num_layers: Optional[int] = Field(default=None, ge=1, le=5, description="Number of LSTM layers")
    sequence_length: Optional[int] = Field(default=None, ge=10, le=120, description="Sequence length")
    experiment_name: Optional[str] = Field(default=None, description="MLflow experiment name")


class TrainResponse(BaseModel):
    """Response after training starts/completes."""
    status: str
    run_id: Optional[str] = None
    message: str
    training_time: Optional[float] = None
    best_val_loss: Optional[float] = None
    test_metrics: Optional[dict] = None


class TrainStatusResponse(BaseModel):
    """Response for training status check."""
    is_training: bool
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_loss: Optional[float] = None
    run_id: Optional[str] = None


# ============== Data ==============

class DataRequest(BaseModel):
    """Request for data retrieval."""
    start_date: Optional[datetime] = Field(default=None, description="Start date filter")
    end_date: Optional[datetime] = Field(default=None, description="End date filter")
    limit: Optional[int] = Field(default=None, ge=1, le=10000, description="Max records to return")
    columns: Optional[List[str]] = Field(default=None, description="Columns to include")


class StockDataItem(BaseModel):
    """Single stock data record."""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class DataResponse(BaseModel):
    """Response with stock data."""
    data: List[StockDataItem]
    total_records: int
    date_range: dict
    columns: List[str]


class DataSummaryResponse(BaseModel):
    """Summary statistics of the data."""
    total_records: int
    date_range: dict
    price_stats: dict
    volume_stats: dict


# ============== Models ==============

class ModelInfo(BaseModel):
    """Information about a model."""
    model_id: str
    checkpoint_path: Optional[str] = None
    run_id: Optional[str] = None
    created_at: Optional[datetime] = None
    config: dict
    metrics: Optional[dict] = None


class ModelsListResponse(BaseModel):
    """List of available models."""
    models: List[ModelInfo]
    current_model: Optional[str] = None
