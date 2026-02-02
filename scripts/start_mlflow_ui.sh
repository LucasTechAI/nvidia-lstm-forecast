#!/bin/bash
# =============================================================================
# NVIDIA LSTM Forecast - Start MLflow UI
# =============================================================================
# Start the MLflow tracking server UI
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  NVIDIA LSTM - MLflow UI Server${NC}"
echo -e "${GREEN}========================================${NC}"

# Default parameters
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-5000}
BACKEND_STORE=${BACKEND_STORE:-"./data/mlruns"}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"./data/mlruns/artifacts"}
USE_DOCKER=${USE_DOCKER:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host     Host to bind to (default: 0.0.0.0)"
            echo "  --port     Port to run on (default: 5000)"
            echo "  --docker   Run using Docker Compose"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

if [ "$USE_DOCKER" = true ]; then
    echo -e "${YELLOW}Starting MLflow UI using Docker Compose...${NC}"
    docker compose up mlflow
else
    echo -e "${YELLOW}Starting MLflow UI locally...${NC}"
    
    # Activate virtual environment if exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Create mlruns directory if it doesn't exist
    mkdir -p "$BACKEND_STORE"
    
    echo "Backend Store: $BACKEND_STORE"
    echo "Artifact Root: $ARTIFACT_ROOT"
    echo ""
    
    echo -e "${GREEN}MLflow UI will be available at: http://localhost:$PORT${NC}"
    echo ""
    
    # Start MLflow server
    mlflow server \
        --backend-store-uri "sqlite:///$BACKEND_STORE/mlflow.db" \
        --default-artifact-root "$ARTIFACT_ROOT" \
        --host "$HOST" \
        --port "$PORT"
fi
