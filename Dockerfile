# Dockerfile for nvidia-lstm-forecast
# Python 3.10 slim image for PyTorch compatibility

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base AS dependencies

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Production Stage
# =============================================================================
FROM dependencies AS production

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/models data/outputs data/mlruns logs

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command - can be overridden
CMD ["python", "-m", "src.training.train"]

# =============================================================================
# Development Stage
# =============================================================================
FROM dependencies AS development

# Install additional dev dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest-cov \
    black \
    isort \
    flake8

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/models data/outputs data/mlruns logs

# Default command for development
CMD ["bash"]

# =============================================================================
# MLflow Server Stage
# =============================================================================
FROM dependencies AS mlflow-server

# Expose MLflow port
EXPOSE 5000

# MLflow server command
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlruns/mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

# =============================================================================
# API Server Stage
# =============================================================================
FROM dependencies AS api-server

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/models/checkpoints data/outputs data/mlruns logs

# Expose API port
EXPOSE 8000

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# API server command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# API with GPU Support
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS api-gpu

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs models mlruns outputs

# Expose MLflow port
EXPOSE 5000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
