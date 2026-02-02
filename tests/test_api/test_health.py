"""
Tests for health endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Test health response has required fields."""
        response = client.get("/health")
        data = response.json()
        
        required_fields = [
            "status",
            "model_loaded", 
            "database_connected",
            "gpu_available",
            "timestamp"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_health_status_values(self, client):
        """Test health status is healthy or degraded."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded"]
    
    def test_health_model_loaded_boolean(self, client):
        """Test model_loaded is a boolean."""
        response = client.get("/health")
        data = response.json()
        
        assert isinstance(data["model_loaded"], bool)


class TestReadinessEndpoint:
    """Tests for the /health/ready endpoint."""
    
    def test_ready_returns_200(self, client):
        """Test readiness endpoint returns 200."""
        response = client.get("/health/ready")
        assert response.status_code == 200
    
    def test_ready_response_structure(self, client):
        """Test readiness response has ready field."""
        response = client.get("/health/ready")
        data = response.json()
        
        assert "ready" in data
        assert isinstance(data["ready"], bool)


class TestLivenessEndpoint:
    """Tests for the /health/live endpoint."""
    
    def test_live_returns_200(self, client):
        """Test liveness endpoint returns 200."""
        response = client.get("/health/live")
        assert response.status_code == 200
    
    def test_live_response_structure(self, client):
        """Test liveness response has alive field."""
        response = client.get("/health/live")
        data = response.json()
        
        assert "alive" in data
        assert data["alive"] is True
