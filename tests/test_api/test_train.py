"""
Tests for train endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app
from src.api.dependencies import model_state


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestTrainStatusEndpoint:
    """Tests for the /train/status endpoint."""
    
    def test_status_returns_200(self, client):
        """Test status endpoint returns 200."""
        response = client.get("/train/status")
        assert response.status_code == 200
    
    def test_status_response_structure(self, client):
        """Test status response structure."""
        response = client.get("/train/status")
        data = response.json()
        
        assert "is_training" in data
        # started_at may not be present if not training
        assert isinstance(data["is_training"], bool)
    
    def test_status_not_training(self, client):
        """Test status when not training."""
        model_state.is_training = False
        
        response = client.get("/train/status")
        data = response.json()
        
        assert data["is_training"] is False


class TestTrainEndpoint:
    """Tests for the /train endpoint."""
    
    def test_train_accepts_empty_body(self, client):
        """Test training endpoint accepts empty body request."""
        # Empty body is valid - all fields are optional
        response = client.post("/train", json={})
        
        # The endpoint should accept the request (not 422 validation error)
        assert response.status_code != 422, "Empty body should be valid"
        
        # Should return 200/202 (training started) or 503 (no model)
        assert response.status_code in [200, 202, 503]
    
    def test_train_epochs_validation_min(self, client):
        """Test epochs must be at least 1."""
        response = client.post("/train", json={"epochs": 0})
        assert response.status_code == 422
    
    def test_train_epochs_validation_max(self, client):
        """Test epochs must be at most 1000."""
        response = client.post("/train", json={"epochs": 2000})
        assert response.status_code == 422
    
    def test_train_batch_size_validation(self, client):
        """Test batch size validation."""
        # Too small
        response = client.post("/train", json={"batch_size": 1})
        assert response.status_code == 422
        
        # Too large
        response = client.post("/train", json={"batch_size": 2000})
        assert response.status_code == 422
    
    def test_train_learning_rate_validation(self, client):
        """Test learning rate validation."""
        response = client.post("/train", json={"learning_rate": -0.01})
        assert response.status_code == 422
        
        response = client.post("/train", json={"learning_rate": 10.0})
        assert response.status_code == 422


class TestTrainSyncEndpoint:
    """Tests for the /train/sync endpoint."""
    
    def test_train_sync_validation(self, client):
        """Test sync training validation."""
        response = client.post("/train/sync", json={"epochs": -5})
        assert response.status_code == 422


class TestStopTrainingEndpoint:
    """Tests for the /train/stop endpoint."""
    
    def test_stop_when_not_training(self, client):
        """Test stop when no training in progress."""
        model_state.is_training = False
        
        response = client.post("/train/stop")
        assert response.status_code == 200
        
        data = response.json()
        assert "No training" in data["message"] or "success" in data["message"].lower()
    
    def test_stop_returns_200(self, client):
        """Test stop endpoint returns 200."""
        response = client.post("/train/stop")
        assert response.status_code == 200


class TestTrainRequestSchemas:
    """Tests for train request schema validation."""
    
    def test_optional_fields(self):
        """Test all fields are optional."""
        from src.api.schemas import TrainRequest
        
        request = TrainRequest()
        assert request.epochs is None
        assert request.batch_size is None
        assert request.learning_rate is None
        assert request.hidden_size is None
    
    def test_valid_request(self):
        """Test valid training request."""
        from src.api.schemas import TrainRequest
        
        request = TrainRequest(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        
        assert request.epochs == 100
        assert request.batch_size == 32
        assert request.learning_rate == 0.001
    
    def test_experiment_name(self):
        """Test experiment name field."""
        from src.api.schemas import TrainRequest
        
        request = TrainRequest(experiment_name="test_experiment")
        assert request.experiment_name == "test_experiment"
