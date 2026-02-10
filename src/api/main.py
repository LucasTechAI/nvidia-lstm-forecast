"""
FastAPI main application.

NVIDIA Stock Price Prediction API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import health_router, predict_router, train_router, data_router
from src.api.dependencies import model_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting NVIDIA Stock Prediction API...")
    
    # Load model on startup
    success = model_state.load_model()
    if success:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded - some endpoints may not work")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="NVIDIA Stock Prediction API",
    description="""REST API for NVIDIA stock price prediction using LSTM neural networks.

## Features

- **Prediction**: Generate stock price forecasts with confidence intervals
- **Inference**: Run predictions on custom input sequences
- **Training**: Train or retrain the LSTM model
- **Data**: Access historical stock data
- **Health**: Monitor API and model status

## Model

Uses a bidirectional LSTM trained on NVIDIA historical stock prices
with Monte Carlo Dropout for uncertainty estimation.

## Usage

1. Check API health: `GET /health`
2. Get predictions: `POST /predict`
3. Access data: `GET /data/historical`
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(train_router)
app.include_router(data_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NVIDIA Stock Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "inference": "/predict/inference",
            "train": "/train",
            "data": "/data",
            "health": "/health"
        }
    }


# For running with: python -m api.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
