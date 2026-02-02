"""
API Routers package.
"""

from src.api.routers.health import router as health_router
from src.api.routers.predict import router as predict_router
from src.api.routers.train import router as train_router
from src.api.routers.data import router as data_router

__all__ = [
    "health_router",
    "predict_router", 
    "train_router",
    "data_router"
]
