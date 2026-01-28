"""Tests for ETL module."""

from src.etl.extractor_nvidia import extract_nvidia_data
from pandas import DataFrame, api
import pytest



class TestExtractNvidiaData:
    """Test cases for NVIDIA data extraction."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_extract_default_period(self):
        """Test extraction with default parameters."""
        df = extract_nvidia_data(period="1mo", interval="1d")
        
        assert df is not None
        assert isinstance(df, DataFrame)
        assert len(df) > 0
    
    @pytest.mark.integration
    def test_extract_columns(self):
        """Test extracted data has expected columns."""
        df = extract_nvidia_data(period="5d", interval="1d")
        
        expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    @pytest.mark.integration
    def test_extract_data_types(self):
        """Test extracted data has correct types."""
        df = extract_nvidia_data(period="5d", interval="1d")
        
        # Numeric columns
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            assert api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"
    
    @pytest.mark.integration
    def test_extract_no_null_values(self):
        """Test extracted data has no null values."""
        df = extract_nvidia_data(period="5d", interval="1d")
        
        # Allow some nulls but not in critical columns
        critical_cols = ["Close", "Volume"]
        for col in critical_cols:
            assert df[col].notna().all(), f"{col} contains null values"
    
    @pytest.mark.integration
    def test_extract_positive_prices(self):
        """Test all prices are positive."""
        df = extract_nvidia_data(period="5d", interval="1d")
        
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            assert (df[col] > 0).all(), f"{col} contains non-positive values"
