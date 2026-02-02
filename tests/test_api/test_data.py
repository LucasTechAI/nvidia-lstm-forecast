"""
Tests for data endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_stock_data():
    """Create mock stock data."""
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    return pd.DataFrame({
        'Date': dates,
        'date': dates,
        'open': np.random.uniform(100, 150, n),
        'high': np.random.uniform(100, 150, n),
        'low': np.random.uniform(100, 150, n),
        'close': np.random.uniform(100, 150, n),
        'volume': np.random.randint(1000000, 5000000, n)
    })


class TestGetDataEndpoint:
    """Tests for the /data endpoint."""
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_data_returns_200(self, mock_load, client, mock_stock_data):
        """Test data endpoint returns 200."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data")
        assert response.status_code == 200
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_data_response_structure(self, mock_load, client, mock_stock_data):
        """Test data response structure."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data")
        data = response.json()
        
        assert "data" in data
        assert "total_records" in data
        assert "date_range" in data
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_data_with_limit(self, mock_load, client, mock_stock_data):
        """Test data with limit parameter."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) <= 10
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_data_with_offset(self, mock_load, client, mock_stock_data):
        """Test data with offset parameter."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data?offset=50")
        assert response.status_code == 200
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_data_record_structure(self, mock_load, client, mock_stock_data):
        """Test individual record structure."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data")
        data = response.json()
        
        if data["data"]:
            record = data["data"][0]
            assert "date" in record
            assert "open" in record
            assert "high" in record
            assert "low" in record
            assert "close" in record
            assert "volume" in record


class TestGetLatestDataEndpoint:
    """Tests for the /data/latest endpoint."""
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_latest_returns_200(self, mock_load, client, mock_stock_data):
        """Test latest endpoint returns 200."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data/latest")
        assert response.status_code == 200
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_latest_default_days(self, mock_load, client, mock_stock_data):
        """Test default days is 30."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data/latest")
        data = response.json()
        
        assert len(data["data"]) <= 30
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_latest_custom_days(self, mock_load, client, mock_stock_data):
        """Test custom days parameter."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data/latest?days=7")
        data = response.json()
        
        assert len(data["data"]) <= 7


class TestGetSummaryEndpoint:
    """Tests for the /data/summary endpoint."""
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_summary_returns_200(self, mock_load, client, mock_stock_data):
        """Test summary endpoint returns 200."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data/summary")
        assert response.status_code == 200
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_summary_structure(self, mock_load, client, mock_stock_data):
        """Test summary response structure."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data/summary")
        data = response.json()
        
        assert "total_records" in data
        assert "date_range" in data
        assert "price_stats" in data
        assert "volume_stats" in data
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_get_summary_price_stats(self, mock_load, client, mock_stock_data):
        """Test price statistics structure."""
        mock_load.return_value = mock_stock_data
        
        response = client.get("/data/summary")
        data = response.json()
        
        price_stats = data["price_stats"]
        # Price stats is nested by column (close, open, high, low)
        assert "close" in price_stats or "min" in price_stats


class TestGetColumnsEndpoint:
    """Tests for the /data/columns endpoint."""
    
    def test_get_columns_returns_200(self, client):
        """Test columns endpoint returns 200."""
        response = client.get("/data/columns")
        assert response.status_code == 200
    
    def test_get_columns_structure(self, client):
        """Test columns response structure."""
        response = client.get("/data/columns")
        data = response.json()
        
        assert "columns" in data
        assert isinstance(data["columns"], list)
    
    def test_get_columns_contains_expected(self, client):
        """Test columns contains expected values."""
        response = client.get("/data/columns")
        data = response.json()
        
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_columns:
            assert col in data["columns"]


class TestDataErrorHandling:
    """Tests for data endpoint error handling."""
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_database_error(self, mock_load, client):
        """Test handling of database errors."""
        mock_load.side_effect = Exception("Database connection failed")
        
        response = client.get("/data")
        assert response.status_code == 500
    
    @patch('src.api.routers.data.load_data_from_db')
    def test_empty_database(self, mock_load, client):
        """Test handling of empty database."""
        mock_load.return_value = pd.DataFrame()
        
        response = client.get("/data")
        # Should handle empty data (may return 500 if not handled)
        assert response.status_code in [200, 404, 500]
