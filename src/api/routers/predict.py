"""
Prediction endpoints for generating forecasts.
"""

from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    PredictRequest, 
    PredictResponse, 
    PredictionItem,
    InferenceRequest,
    InferenceResponse
)
from src.api.dependencies import ModelState, get_model_state
from src.etl.preprocessing import load_data_from_db
from src.config import settings

router = APIRouter(prefix="/predict", tags=["Prediction"])


def get_forecast_dates(last_date: pd.Timestamp, horizon: int) -> List[datetime]:
    """Generate trading day dates (skip weekends)."""
    forecast_dates = []
    current_date = pd.Timestamp(last_date)
    
    while len(forecast_dates) < horizon:
        current_date = current_date + pd.Timedelta(days=1)
        if current_date.weekday() < 5:  # Skip weekends
            forecast_dates.append(current_date.to_pydatetime())
    
    return forecast_dates


def generate_forecast_with_uncertainty(
    model: torch.nn.Module,
    initial_sequence: torch.Tensor,
    horizon: int,
    n_samples: int,
    device: str
) -> tuple:
    """Generate forecast with Monte Carlo Dropout for uncertainty."""
    model.train()  # Enable dropout
    
    all_predictions = []
    
    for _ in range(n_samples):
        sequence = initial_sequence.clone()
        predictions = []
        
        for _ in range(horizon):
            with torch.no_grad():
                pred = model(sequence)
            predictions.append(pred.cpu().numpy().flatten()[0])
            
            # Sliding window update
            new_input = pred.unsqueeze(0)
            sequence = torch.cat([sequence[:, 1:, :], new_input], dim=1)
        
        all_predictions.append(predictions)
    
    model.eval()
    all_predictions = np.array(all_predictions)
    
    mean_preds = np.mean(all_predictions, axis=0)
    std_preds = np.std(all_predictions, axis=0)
    
    return mean_preds, std_preds


@router.post("", response_model=PredictResponse)
@router.post("/", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    state: ModelState = Depends(get_model_state)
) -> PredictResponse:
    """
    Generate stock price predictions for the next N days.
    
    Uses the best trained LSTM model with optional Monte Carlo Dropout
    for uncertainty estimation.
    """
    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call /health to check status."
        )
    
    try:
        # Load historical data
        df = load_data_from_db(start_year=2017)
        df['date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Prepare sequence
        sequence_length = state.model_config.get('sequence_length', 60)
        close_prices = df['close'].values.reshape(-1, 1)
        normalized = state.scaler.transform(close_prices)
        
        last_sequence = normalized[-sequence_length:]
        sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(state.device)
        
        # Generate predictions
        if request.with_uncertainty:
            mean_preds, std_preds = generate_forecast_with_uncertainty(
                state.model,
                sequence_tensor,
                request.horizon,
                request.n_samples,
                state.device
            )
            
            z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(request.confidence_level, 1.96)
            lower = mean_preds - z_score * std_preds
            upper = mean_preds + z_score * std_preds
            
            # Inverse transform
            predictions_real = state.scaler.inverse_transform(mean_preds.reshape(-1, 1)).flatten()
            lower_real = state.scaler.inverse_transform(lower.reshape(-1, 1)).flatten()
            upper_real = state.scaler.inverse_transform(upper.reshape(-1, 1)).flatten()
        else:
            state.model.eval()
            sequence = sequence_tensor.clone()
            predictions = []
            
            for _ in range(request.horizon):
                with torch.no_grad():
                    pred = state.model(sequence)
                predictions.append(pred.cpu().numpy().flatten()[0])
                new_input = pred.unsqueeze(0)
                sequence = torch.cat([sequence[:, 1:, :], new_input], dim=1)
            
            predictions = np.array(predictions)
            predictions_real = state.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            lower_real = None
            upper_real = None
        
        # Generate dates
        last_date = df['date'].iloc[-1]
        forecast_dates = get_forecast_dates(last_date, request.horizon)
        
        # Build response
        prediction_items = []
        for i, date in enumerate(forecast_dates):
            item = PredictionItem(
                date=date,
                predicted_price=float(predictions_real[i]),
                lower_bound=float(lower_real[i]) if lower_real is not None else None,
                upper_bound=float(upper_real[i]) if upper_real is not None else None
            )
            prediction_items.append(item)
        
        return PredictResponse(
            predictions=prediction_items,
            last_known_price=float(df['close'].iloc[-1]),
            last_known_date=df['date'].iloc[-1].to_pydatetime(),
            forecast_horizon=request.horizon,
            model_info=state.model_config,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/inference", response_model=InferenceResponse)
async def inference(
    request: InferenceRequest,
    state: ModelState = Depends(get_model_state)
) -> InferenceResponse:
    """
    Perform inference on a custom input sequence.
    
    Use this endpoint when you want to provide your own price sequence
    for prediction instead of using historical data.
    """
    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Normalize input
        input_array = np.array(request.sequence).reshape(-1, 1)
        normalized = state.scaler.transform(input_array)
        
        # Prepare tensor
        sequence_tensor = torch.FloatTensor(normalized).unsqueeze(0).to(state.device)
        
        # Generate predictions
        state.model.eval()
        sequence = sequence_tensor.clone()
        predictions = []
        
        for _ in range(request.steps):
            with torch.no_grad():
                pred = state.model(sequence)
            predictions.append(pred.cpu().numpy().flatten()[0])
            new_input = pred.unsqueeze(0)
            sequence = torch.cat([sequence[:, 1:, :], new_input], dim=1)
        
        # Inverse transform
        predictions = np.array(predictions)
        predictions_real = state.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return InferenceResponse(
            predictions=predictions_real.tolist(),
            input_length=len(request.sequence),
            generated_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )
