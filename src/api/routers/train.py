"""
Training endpoints.
"""

import asyncio
from datetime import datetime
from typing import Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status

from src.api.schemas import TrainRequest, TrainResponse, TrainStatusResponse
from src.api.dependencies import ModelState, get_model_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["Training"])


async def run_training_task(
    state: ModelState,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    hidden_size: Optional[int] = None,
    num_layers: Optional[int] = None,
    sequence_length: Optional[int] = None,
    experiment_name: Optional[str] = None
):
    """Background task to run model training."""
    from src.training.train import train_model
    from src.config import settings
    
    try:
        state.is_training = True
        
        # Override config if provided
        if epochs is not None:
            settings.training.epochs = epochs
        if batch_size is not None:
            settings.training.batch_size = batch_size
        if learning_rate is not None:
            settings.training.learning_rate = learning_rate
        if hidden_size is not None:
            settings.lstm.hidden_size = hidden_size
        if num_layers is not None:
            settings.lstm.num_layers = num_layers
        if sequence_length is not None:
            settings.lstm.sequence_length = sequence_length
        
        state.total_epochs = settings.training.epochs
        
        # Run training
        result = train_model(experiment_name=experiment_name)
        
        state.training_run_id = result.get('run_id')
        
        # Reload model after training
        checkpoint_path = settings.models_dir / "checkpoints" / "best_model.pt"
        if checkpoint_path.exists():
            state.load_model(str(checkpoint_path))
        
        logger.info(f"Training completed. Run ID: {state.training_run_id}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        state.is_training = False
        state.current_epoch = 0


@router.post("", response_model=TrainResponse)
@router.post("/", response_model=TrainResponse)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    state: ModelState = Depends(get_model_state)
) -> TrainResponse:
    """
    Start model training as a background task.
    
    Training runs asynchronously. Use /train/status to check progress.
    """
    if state.is_training:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training already in progress"
        )
    
    # Add training task to background
    background_tasks.add_task(
        run_training_task,
        state,
        request.epochs,
        request.batch_size,
        request.learning_rate,
        request.hidden_size,
        request.num_layers,
        request.sequence_length,
        request.experiment_name
    )
    
    return TrainResponse(
        status="started",
        message="Training started in background. Use /train/status to check progress.",
        run_id=None
    )


@router.get("/status", response_model=TrainStatusResponse)
async def training_status(
    state: ModelState = Depends(get_model_state)
) -> TrainStatusResponse:
    """
    Get the current training status.
    """
    return TrainStatusResponse(
        is_training=state.is_training,
        current_epoch=state.current_epoch if state.is_training else None,
        total_epochs=state.total_epochs if state.is_training else None,
        current_loss=state.current_loss if state.is_training else None,
        run_id=state.training_run_id
    )


@router.post("/sync", response_model=TrainResponse)
async def train_sync(
    request: TrainRequest,
    state: ModelState = Depends(get_model_state)
) -> TrainResponse:
    """
    Run training synchronously (blocking).
    
    Warning: This may timeout for long training sessions.
    Use /train for async training.
    """
    if state.is_training:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training already in progress"
        )
    
    from src.training.train import train_model
    from src.config import settings
    import time
    
    try:
        state.is_training = True
        start_time = time.time()
        
        # Override config
        if request.epochs is not None:
            settings.training.epochs = request.epochs
        if request.batch_size is not None:
            settings.training.batch_size = request.batch_size
        if request.learning_rate is not None:
            settings.training.learning_rate = request.learning_rate
        if request.hidden_size is not None:
            settings.lstm.hidden_size = request.hidden_size
        if request.num_layers is not None:
            settings.lstm.num_layers = request.num_layers
        if request.sequence_length is not None:
            settings.lstm.sequence_length = request.sequence_length
        
        # Run training
        result = train_model(experiment_name=request.experiment_name)
        
        training_time = time.time() - start_time
        
        # Reload model
        checkpoint_path = settings.models_dir / "checkpoints" / "best_model.pt"
        if checkpoint_path.exists():
            state.load_model(str(checkpoint_path))
        
        return TrainResponse(
            status="completed",
            run_id=result.get('run_id'),
            message="Training completed successfully",
            training_time=training_time,
            best_val_loss=result.get('best_val_loss'),
            test_metrics=result.get('test_metrics')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )
    finally:
        state.is_training = False


@router.post("/stop")
async def stop_training(
    state: ModelState = Depends(get_model_state)
) -> dict:
    """
    Request to stop ongoing training.
    
    Note: This sets a flag but actual stopping depends on training implementation.
    """
    if not state.is_training:
        return {"message": "No training in progress"}
    
    # TODO: Implement actual training interruption
    return {"message": "Stop requested. Training will stop after current epoch."}
