"""
Tests for API schemas.
"""

import pytest
from pydantic import ValidationError
from datetime import datetime

from src.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionItem,
    InferenceRequest,
    InferenceResponse,
    TrainRequest,
    TrainResponse,
    DataRequest,
    DataResponse,
    StockDataItem
)


class TestHealthResponse:
    """Tests for HealthResponse schema."""
    
    def test_valid_response(self):
        """Test valid health response."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            database_connected=True,
            gpu_available=False,
            timestamp=datetime.now()
        )
        
        assert response.status == "healthy"
        assert response.model_loaded is True
    
    def test_degraded_status(self):
        """Test degraded status is valid."""
        response = HealthResponse(
            status="degraded",
            model_loaded=False,
            database_connected=True,
            gpu_available=False,
            timestamp=datetime.now()
        )
        
        assert response.status == "degraded"


class TestPredictRequest:
    """Tests for PredictRequest schema."""
    
    def test_default_values(self):
        """Test default values."""
        request = PredictRequest()
        
        assert request.horizon == 30
        assert request.with_uncertainty is True
        assert request.n_samples == 100
        assert request.confidence_level == 0.95
    
    def test_custom_horizon(self):
        """Test custom horizon."""
        request = PredictRequest(horizon=7)
        assert request.horizon == 7
    
    def test_horizon_validation_min(self):
        """Test horizon minimum validation."""
        with pytest.raises(ValidationError):
            PredictRequest(horizon=0)
    
    def test_horizon_validation_max(self):
        """Test horizon maximum validation."""
        with pytest.raises(ValidationError):
            PredictRequest(horizon=500)
    
    def test_confidence_level_validation(self):
        """Test confidence level validation."""
        with pytest.raises(ValidationError):
            PredictRequest(confidence_level=1.5)
        
        with pytest.raises(ValidationError):
            PredictRequest(confidence_level=0.1)
    
    def test_n_samples_validation(self):
        """Test n_samples validation."""
        with pytest.raises(ValidationError):
            PredictRequest(n_samples=5)  # Min is 10


class TestInferenceRequest:
    """Tests for InferenceRequest schema."""
    
    def test_valid_request(self):
        """Test valid inference request."""
        request = InferenceRequest(
            sequence=[1.0, 2.0, 3.0, 4.0, 5.0],
            steps=3
        )
        
        assert len(request.sequence) == 5
        assert request.steps == 3
    
    def test_empty_sequence_fails(self):
        """Test empty sequence fails validation."""
        with pytest.raises(ValidationError):
            InferenceRequest(sequence=[], steps=1)
    
    def test_steps_validation(self):
        """Test steps validation."""
        with pytest.raises(ValidationError):
            InferenceRequest(sequence=[1.0, 2.0], steps=0)
    
    def test_steps_max_validation(self):
        """Test steps max validation (30)."""
        with pytest.raises(ValidationError):
            InferenceRequest(sequence=[1.0, 2.0], steps=50)


class TestTrainRequest:
    """Tests for TrainRequest schema."""
    
    def test_all_optional(self):
        """Test all fields are optional."""
        request = TrainRequest()
        
        assert request.epochs is None
        assert request.batch_size is None
        assert request.learning_rate is None
    
    def test_valid_request(self):
        """Test valid training request."""
        request = TrainRequest(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            hidden_size=128,
            num_layers=2,
            sequence_length=60,
            experiment_name="test"
        )
        
        assert request.epochs == 100
        assert request.batch_size == 32
    
    def test_epochs_validation(self):
        """Test epochs validation."""
        with pytest.raises(ValidationError):
            TrainRequest(epochs=0)
        
        with pytest.raises(ValidationError):
            TrainRequest(epochs=1000)  # Max is 500
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        with pytest.raises(ValidationError):
            TrainRequest(batch_size=1)  # Min is 8
        
        with pytest.raises(ValidationError):
            TrainRequest(batch_size=500)  # Max is 256
    
    def test_learning_rate_validation(self):
        """Test learning rate validation."""
        with pytest.raises(ValidationError):
            TrainRequest(learning_rate=0)  # Min is 1e-6
        
        with pytest.raises(ValidationError):
            TrainRequest(learning_rate=1.0)  # Max is 0.1


class TestStockDataItem:
    """Tests for StockDataItem schema."""
    
    def test_valid_datapoint(self):
        """Test valid stock data point."""
        point = StockDataItem(
            date=datetime(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000
        )
        
        assert point.close == 102.0
        assert point.volume == 1000000


class TestDataResponse:
    """Tests for DataResponse schema."""
    
    def test_valid_response(self):
        """Test valid data response."""
        response = DataResponse(
            data=[
                StockDataItem(
                    date=datetime(2024, 1, 1),
                    open=100.0,
                    high=105.0,
                    low=95.0,
                    close=102.0,
                    volume=1000000
                )
            ],
            total_records=1,
            date_range={"start": "2024-01-01", "end": "2024-01-01"},
            columns=["open", "high", "low", "close", "volume"]
        )
        
        assert len(response.data) == 1
        assert response.total_records == 1
    
    def test_empty_data(self):
        """Test empty data response."""
        response = DataResponse(
            data=[],
            total_records=0,
            date_range={"start": None, "end": None},
            columns=[]
        )
        
        assert len(response.data) == 0


class TestPredictionItem:
    """Tests for PredictionItem schema."""
    
    def test_valid_item(self):
        """Test valid prediction item."""
        item = PredictionItem(
            date=datetime(2024, 1, 1),
            predicted_price=150.0,
            lower_bound=145.0,
            upper_bound=155.0
        )
        
        assert item.predicted_price == 150.0
        assert item.lower_bound == 145.0
        assert item.upper_bound == 155.0
    
    def test_optional_bounds(self):
        """Test optional bounds."""
        item = PredictionItem(
            date=datetime(2024, 1, 1),
            predicted_price=150.0
        )
        
        assert item.lower_bound is None
        assert item.upper_bound is None


class TestPredictResponse:
    """Tests for PredictResponse schema."""
    
    def test_valid_response(self):
        """Test valid predict response."""
        response = PredictResponse(
            predictions=[
                PredictionItem(
                    date=datetime(2024, 1, 1),
                    predicted_price=150.0
                )
            ],
            last_known_price=148.0,
            last_known_date=datetime(2023, 12, 31),
            forecast_horizon=30,
            model_info={"hidden_size": 128}
        )
        
        assert len(response.predictions) == 1
        assert response.forecast_horizon == 30


class TestTrainResponse:
    """Tests for TrainResponse schema."""
    
    def test_valid_response(self):
        """Test valid train response."""
        response = TrainResponse(
            status="completed",
            message="Training completed successfully",
            run_id="abc123"
        )
        
        assert response.status == "completed"
        assert response.run_id == "abc123"
    
    def test_with_metrics(self):
        """Test train response with metrics."""
        response = TrainResponse(
            status="completed",
            message="Training completed",
            run_id="abc123",
            training_time=120.5,
            best_val_loss=0.0015,
            test_metrics={"rmse": 2.5, "mae": 1.8}
        )
        
        assert response.training_time == 120.5
        assert response.best_val_loss == 0.0015


class TestInferenceResponse:
    """Tests for InferenceResponse schema."""
    
    def test_valid_response(self):
        """Test valid inference response."""
        response = InferenceResponse(
            predictions=[150.0, 151.0, 152.0],
            input_length=60
        )
        
        assert len(response.predictions) == 3
        assert response.input_length == 60
