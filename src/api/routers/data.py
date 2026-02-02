"""
Data retrieval endpoints.
"""

from datetime import datetime
from typing import Optional, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import (
    DataRequest,
    DataResponse,
    StockDataItem,
    DataSummaryResponse
)
from src.etl.preprocessing import load_data_from_db
from src.config import settings

router = APIRouter(prefix="/data", tags=["Data"])


@router.get("", response_model=DataResponse)
@router.get("/", response_model=DataResponse)
async def get_data(
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Max records"),
    start_year: int = Query(2017, description="Start year for data loading")
) -> DataResponse:
    """
    Retrieve historical stock data.
    
    Supports filtering by date range and limiting results.
    """
    try:
        df = load_data_from_db(start_year=start_year)
        df['date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Apply filters
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        if limit:
            df = df.tail(limit)
        
        # Convert to response format
        data_items = []
        for _, row in df.iterrows():
            item = StockDataItem(
                date=row['date'].to_pydatetime(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            )
            data_items.append(item)
        
        return DataResponse(
            data=data_items,
            total_records=len(data_items),
            date_range={
                "start": df['date'].min().isoformat() if len(df) > 0 else None,
                "end": df['date'].max().isoformat() if len(df) > 0 else None
            },
            columns=["date", "open", "high", "low", "close", "volume"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve data: {str(e)}"
        )


@router.get("/latest", response_model=DataResponse)
async def get_latest_data(
    days: int = Query(30, ge=1, le=365, description="Number of recent days")
) -> DataResponse:
    """
    Get the most recent N days of stock data.
    """
    try:
        df = load_data_from_db(start_year=2017)
        df['date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.tail(days)
        
        data_items = []
        for _, row in df.iterrows():
            item = StockDataItem(
                date=row['date'].to_pydatetime(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            )
            data_items.append(item)
        
        return DataResponse(
            data=data_items,
            total_records=len(data_items),
            date_range={
                "start": df['date'].min().isoformat() if len(df) > 0 else None,
                "end": df['date'].max().isoformat() if len(df) > 0 else None
            },
            columns=["date", "open", "high", "low", "close", "volume"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve data: {str(e)}"
        )


@router.get("/summary", response_model=DataSummaryResponse)
async def get_data_summary(
    start_year: int = Query(2017, description="Start year for data")
) -> DataSummaryResponse:
    """
    Get summary statistics of the stock data.
    """
    try:
        df = load_data_from_db(start_year=start_year)
        df['date'] = pd.to_datetime(df['Date'])
        
        return DataSummaryResponse(
            total_records=len(df),
            date_range={
                "start": df['date'].min().isoformat(),
                "end": df['date'].max().isoformat(),
                "trading_days": len(df)
            },
            price_stats={
                "close": {
                    "min": float(df['close'].min()),
                    "max": float(df['close'].max()),
                    "mean": float(df['close'].mean()),
                    "std": float(df['close'].std()),
                    "latest": float(df['close'].iloc[-1])
                },
                "open": {
                    "min": float(df['open'].min()),
                    "max": float(df['open'].max()),
                    "mean": float(df['open'].mean())
                },
                "high": {
                    "min": float(df['high'].min()),
                    "max": float(df['high'].max()),
                    "mean": float(df['high'].mean())
                },
                "low": {
                    "min": float(df['low'].min()),
                    "max": float(df['low'].max()),
                    "mean": float(df['low'].mean())
                }
            },
            volume_stats={
                "min": int(df['volume'].min()),
                "max": int(df['volume'].max()),
                "mean": float(df['volume'].mean()),
                "total": int(df['volume'].sum())
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get summary: {str(e)}"
        )


@router.get("/columns")
async def get_columns() -> dict:
    """
    Get available columns in the dataset.
    """
    return {
        "columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
        "target_column": "Close",
        "date_column": "Date"
    }
