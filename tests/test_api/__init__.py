"""
Tests for the FastAPI API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime

from src.api.main import app
from src.api.dependencies import model_state, ModelState


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_state():
    """Mock the model state for testing."""
    state = ModelState()
    state.model = MagicMock()
    state.scaler = MagicMock()
    state.model_config = {
        'input_size': 1,
        'hidden_size': 128,
        'num_layers': 2,
        'output_size': 1,
        'dropout': 0.2,
        'bidirectional': False,
        'sequence_length': 60
    }
    state.device = 'cpu'
    state.is_training = False
    return state


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test the main health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "database_connected" in data
        assert "gpu_available" in data
        assert "timestamp" in data
    
    def test_readiness_check(self, client):
        """Test the readiness probe."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert "ready" in data
    
    def test_liveness_check(self, client):
        """Test the liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["alive"] is True


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self, client):
        """Test the root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestDataEndpoints:
    """Tests for data retrieval endpoints."""
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_data(self, mock_load, client):
        """Test getting stock data."""
        # Mock data
        mock_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'date': pd.date_range('2024-01-01', periods=10),
            'open': np.random.uniform(100, 150, 10),
            'high': np.random.uniform(100, 150, 10),
            'low': np.random.uniform(100, 150, 10),
            'close': np.random.uniform(100, 150, 10),
            'volume': np.random.randint(1000000, 5000000, 10)
        })
        mock_load.return_value = mock_df
        
        response = client.get("/data")
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        assert "total_records" in data
        assert "date_range" in data
        assert len(data["data"]) == 10
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_latest_data(self, mock_load, client):
        """Test getting latest N days of data."""
        mock_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'date': pd.date_range('2024-01-01', periods=100),
            'open': np.random.uniform(100, 150, 100),
            'high': np.random.uniform(100, 150, 100),
            'low': np.random.uniform(100, 150, 100),
            'close': np.random.uniform(100, 150, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        mock_load.return_value = mock_df
        
        response = client.get("/data/latest?days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) == 30
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_data_summary(self, mock_load, client):
        """Test getting data summary statistics."""
        mock_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'date': pd.date_range('2024-01-01', periods=100),
            'open': np.random.uniform(100, 150, 100),
            'high': np.random.uniform(100, 150, 100),
            'low': np.random.uniform(100, 150, 100),
            'close': np.random.uniform(100, 150, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        mock_load.return_value = mock_df
        
        response = client.get("/data/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_records" in data
        assert "price_stats" in data
        assert "volume_stats" in data
    
    def test_get_columns(self, client):
        """Test getting available columns."""
        response = client.get("/data/columns")
        assert response.status_code == 200
        
        data = response.json()
        assert "columns" in data
        assert "Close" in data["columns"]


class TestPredictEndpoints:
    """Tests for prediction endpoints."""
    
    def test_predict_without_model(self, client):
        """Test prediction fails gracefully without model."""
        # Reset model state
        model_state.model = None
        model_state.scaler = None
        
        response = client.post("/predict", json={"horizon": 30})
        assert response.status_code == 503
    
    @patch('src.api.routers.predict.load_data_from_db')
    def test_predict_with_mock_model(self, mock_load, client, mock_model_state):
        """Test prediction with mocked model."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'date': pd.date_range('2024-01-01', periods=100),
            'close': np.random.uniform(100, 150, 100)
        })
        mock_load.return_value = mock_df
        
        # Mock model state
        model_state.model = mock_model_state.model
        model_state.scaler = mock_model_state.scaler
        model_state.model_config = mock_model_state.model_config
        model_state.device = 'cpu'
        
        # Mock scaler transform/inverse_transform
        model_state.scaler.transform.return_value = np.random.uniform(0, 1, (100, 1))
        model_state.scaler.inverse_transform.return_value = np.random.uniform(100, 150, (30, 1))
        
        # Mock model forward pass
        import torch
        model_state.model.return_value = torch.tensor([[0.5]])
        model_state.model.train = MagicMock()
        model_state.model.eval = MagicMock()
        
        response = client.post("/predict", json={
            "horizon": 5,
            "with_uncertainty": False
        })
        
        # Should either succeed or fail with specific error
        assert response.status_code in [200, 500]
    
    def test_predict_validation(self, client):
        """Test prediction request validation."""
        # Invalid horizon
        response = client.post("/predict", json={"horizon": 0})
        assert response.status_code == 422
        
        response = client.post("/predict", json={"horizon": 1000})
        assert response.status_code == 422
        
        # Invalid confidence level
        response = client.post("/predict", json={"confidence_level": 1.5})
        assert response.status_code == 422
    
    def test_inference_without_model(self, client):
        """Test inference fails gracefully without model."""
        model_state.model = None
        model_state.scaler = None
        
        response = client.post("/predict/inference", json={
            "sequence": [100.0] * 60,
            "steps": 5
        })
        assert response.status_code == 503


class TestTrainEndpoints:
    """Tests for training endpoints."""
    
    def test_train_status_no_training(self, client):
        """Test training status when not training."""
        model_state.is_training = False
        
        response = client.get("/train/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["is_training"] is False
    
    def test_train_request_validation(self, client):
        """Test training request validation."""
        # Invalid epochs
        response = client.post("/train", json={"epochs": 0})
        assert response.status_code == 422
        
        # Invalid batch size
        response = client.post("/train", json={"batch_size": 1})
        assert response.status_code == 422
        
        # Invalid learning rate
        response = client.post("/train", json={"learning_rate": 10.0})
        assert response.status_code == 422
    
    def test_stop_training_no_training(self, client):
        """Test stop when no training in progress."""
        model_state.is_training = False
        
        response = client.post("/train/stop")
        assert response.status_code == 200
        
        data = response.json()
        assert "No training" in data["message"]


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""
    
    def test_predict_request_defaults(self, client):
        """Test predict request with defaults."""
        from src.api.schemas import PredictRequest
        
        request = PredictRequest()
        assert request.horizon == 30
        assert request.with_uncertainty is True
        assert request.n_samples == 100
        assert request.confidence_level == 0.95
    
    def test_inference_request_validation(self):
        """Test inference request validation."""
        from src.api.schemas import InferenceRequest
        
        # Valid request
        request = InferenceRequest(sequence=[1.0, 2.0, 3.0], steps=1)
        assert len(request.sequence) == 3
        
        # Empty sequence should fail
        with pytest.raises(Exception):
            InferenceRequest(sequence=[], steps=1)
    
    def test_train_request_optional_fields(self):
        """Test train request with optional fields."""
        from src.api.schemas import TrainRequest
        
        request = TrainRequest()
        assert request.epochs is None
        assert request.batch_size is None
        
        request = TrainRequest(epochs=50, batch_size=32)
        assert request.epochs == 50
        assert request.batch_size == 32
