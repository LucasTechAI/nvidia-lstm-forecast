from src.api.schemas.health_schema import HealthResponse, Health
from src.api.services.health_service import check_health
from src.api.utils.jwt_handler import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from logging import getLogger, basicConfig, INFO

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
router = APIRouter(prefix="/api/v1/health", tags=["Health"])
logger = getLogger(__name__)
basicConfig(level=INFO, format=FORMAT)

@router.get("/", **Health.docs)
def health(current_user: dict = Depends(get_current_user)) -> HealthResponse:
    """
    Health check endpoint to verify if the API is running.
    Returns:
        dict: A dictionary indicating the health status of the API.
    """
    try:
        health = check_health()
        if not health:
            raise HTTPException(status_code=404, detail="Health check failed")
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error(f"Error during health check: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")