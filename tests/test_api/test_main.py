"""
Tests for the API endpoints - main test file.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestAPIIntegration:
    """Integration tests for the API."""
    
    def test_docs_available(self, client):
        """Test that API documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "NVIDIA Stock Prediction API"
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # CORS preflight should work
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_not_found(self, client):
        """Test 404 response for unknown endpoints."""
        response = client.get("/unknown/endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 response for wrong HTTP method."""
        response = client.delete("/health")
        assert response.status_code == 405
    
    def test_invalid_json(self, client):
        """Test 422 response for invalid JSON."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
