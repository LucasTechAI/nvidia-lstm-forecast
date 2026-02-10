"""
Health check endpoint.
"""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
import torch

from src.api.schemas import HealthResponse
from src.api.dependencies import ModelState, get_model_state
from src.config import settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
@router.get("/", response_model=HealthResponse)
async def health_check(state: ModelState = Depends(get_model_state)) -> HealthResponse:
    """
    Check the health status of the API.
    
    Returns:
        HealthResponse with status of all components.
    """
    # Check database
    db_connected = False
    try:
        db_path = Path(settings.database_path)
        db_connected = db_path.exists()
    except Exception:
        pass
    
    # Check GPU
    gpu_available = torch.cuda.is_available()
    
    # Check model
    model_loaded = state.is_ready
    
    # Overall status
    status = "healthy" if (model_loaded and db_connected) else "degraded"
    if not db_connected:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        database_connected=db_connected,
        gpu_available=gpu_available,
        timestamp=datetime.now()
    )


@router.get("/ready")
async def readiness_check(state: ModelState = Depends(get_model_state)) -> dict:
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if ready to serve requests, 503 otherwise.
    """
    if state.is_ready:
        return {"ready": True}
    return {"ready": False, "reason": "Model not loaded"}


@router.get("/live")
async def liveness_check() -> dict:
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if the service is alive.
    """
    return {"alive": True}
