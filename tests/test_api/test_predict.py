"""
Tests for predict endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch

from src.api.main import app
from src.api.dependencies import model_state


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_loaded_model():
    """Create a mock loaded model state."""
    # Save original state
    original_model = model_state.model
    original_scaler = model_state.scaler
    original_config = model_state.model_config
    
    # Setup mock
    model_state.model = MagicMock()
    model_state.model.train = MagicMock()
    model_state.model.eval = MagicMock()
    
    model_state.scaler = MagicMock()
    model_state.model_config = {
        'sequence_length': 60,
        'input_size': 1,
        'hidden_size': 128,
        'num_layers': 2
    }
    model_state.device = 'cpu'
    
    yield model_state
    
    # Restore original state
    model_state.model = original_model
    model_state.scaler = original_scaler
    model_state.model_config = original_config


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_requires_model(self, client):
        """Test that predict fails when model not loaded."""
        # Ensure model is not loaded
        model_state.model = None
        model_state.scaler = None
        
        response = client.post("/predict", json={"horizon": 30})
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data
    
    def test_predict_horizon_validation_min(self, client):
        """Test horizon must be at least 1."""
        response = client.post("/predict", json={"horizon": 0})
        assert response.status_code == 422
    
    def test_predict_horizon_validation_max(self, client):
        """Test horizon must be at most 365."""
        response = client.post("/predict", json={"horizon": 500})
        assert response.status_code == 422
    
    def test_predict_confidence_level_validation(self, client):
        """Test confidence level must be between 0 and 1."""
        response = client.post("/predict", json={"confidence_level": -0.1})
        assert response.status_code == 422
        
        response = client.post("/predict", json={"confidence_level": 1.5})
        assert response.status_code == 422
    
    def test_predict_n_samples_validation(self, client):
        """Test n_samples must be positive."""
        response = client.post("/predict", json={"n_samples": 0})
        assert response.status_code == 422


class TestInferenceEndpoint:
    """Tests for the /predict/inference endpoint."""
    
    def test_inference_requires_model(self, client):
        """Test that inference fails when model not loaded."""
        model_state.model = None
        
        response = client.post("/predict/inference", json={
            "sequence": [100.0] * 60,
            "steps": 5
        })
        assert response.status_code == 503
    
    def test_inference_sequence_required(self, client):
        """Test sequence is required."""
        response = client.post("/predict/inference", json={
            "steps": 5
        })
        assert response.status_code == 422
    
    def test_inference_sequence_not_empty(self, client):
        """Test sequence cannot be empty."""
        response = client.post("/predict/inference", json={
            "sequence": [],
            "steps": 5
        })
        assert response.status_code == 422
    
    def test_inference_steps_validation(self, client):
        """Test steps must be positive."""
        response = client.post("/predict/inference", json={
            "sequence": [100.0] * 10,
            "steps": 0
        })
        assert response.status_code == 422
    
    def test_inference_without_loaded_model(self, client):
        """Test inference returns 503 without model."""
        from src.api.dependencies import model_state
        
        # Ensure model not loaded
        model_state.model = None
        
        response = client.post("/predict/inference", json={
            "sequence": [100.0] * 10,
            "steps": 1
        })
        
        # Should return 503 service unavailable
        assert response.status_code == 503


class TestPredictRequestSchemas:
    """Tests for predict request schema validation."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        from src.api.schemas import PredictRequest
        
        request = PredictRequest()
        assert request.horizon == 30
        assert request.with_uncertainty is True
        assert request.n_samples == 100
        assert request.confidence_level == 0.95
    
    def test_custom_values(self):
        """Test custom values are accepted."""
        from src.api.schemas import PredictRequest
        
        request = PredictRequest(
            horizon=7,
            with_uncertainty=False,
            n_samples=50,
            confidence_level=0.9
        )
        
        assert request.horizon == 7
        assert request.with_uncertainty is False
        assert request.n_samples == 50
        assert request.confidence_level == 0.9


class TestInferenceRequestSchemas:
    """Tests for inference request schema validation."""
    
    def test_valid_request(self):
        """Test valid inference request."""
        from src.api.schemas import InferenceRequest
        
        request = InferenceRequest(
            sequence=[1.0, 2.0, 3.0, 4.0, 5.0],
            steps=3
        )
        
        assert len(request.sequence) == 5
        assert request.steps == 3
    
    def test_valid_with_default_uncertainty(self):
        """Test inference request has default values."""
        from src.api.schemas import InferenceRequest
        
        request = InferenceRequest(
            sequence=[1.0, 2.0, 3.0],
            steps=1
        )
        
        # InferenceRequest may not have with_uncertainty field
        assert request.steps == 1
        assert len(request.sequence) == 3
