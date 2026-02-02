#!/bin/bash
# =============================================================================
# Script to run the FastAPI server
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo -e "${GREEN}🚀 Starting NVIDIA Stock Prediction API...${NC}"
echo -e "${YELLOW}   Host: ${HOST}${NC}"
echo -e "${YELLOW}   Port: ${PORT}${NC}"
echo -e "${YELLOW}   Reload: ${RELOAD}${NC}"
echo -e "${YELLOW}   Log Level: ${LOG_LEVEL}${NC}"
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API
if [ "$RELOAD" = "true" ]; then
    uvicorn api.main:app --host "$HOST" --port "$PORT" --reload --log-level "$LOG_LEVEL"
else
    uvicorn api.main:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
