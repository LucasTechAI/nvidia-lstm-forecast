#!/bin/bash
# =============================================================================
# NVIDIA LSTM Forecast - Prediction Script
# =============================================================================
# Generate stock price forecasts using trained model
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  NVIDIA LSTM - Stock Price Forecast${NC}"
echo -e "${GREEN}========================================${NC}"

# Default parameters
HORIZON=${HORIZON:-30}
RUN_ID=${RUN_ID:-""}
CHECKPOINT=${CHECKPOINT:-""}
SCALER=${SCALER:-""}
NO_UNCERTAINTY=${NO_UNCERTAINTY:-false}
NO_SAVE=${NO_SAVE:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --scaler)
            SCALER="$2"
            shift 2
            ;;
        --no-uncertainty)
            NO_UNCERTAINTY=true
            shift
            ;;
        --no-save)
            NO_SAVE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --horizon        Forecast horizon in days (default: 30)"
            echo "  --run-id         MLflow run ID to load model from"
            echo "  --checkpoint     Path to model checkpoint file"
            echo "  --scaler         Path to scaler file"
            echo "  --no-uncertainty Disable uncertainty estimation"
            echo "  --no-save        Do not save results to files"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Prediction Parameters:${NC}"
echo "  Forecast Horizon: $HORIZON days"
echo "  MLflow Run ID:    ${RUN_ID:-auto-detect}"
echo "  Checkpoint:       ${CHECKPOINT:-auto-detect}"
echo "  Uncertainty:      $([ "$NO_UNCERTAINTY" = true ] && echo "disabled" || echo "enabled")"
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo -e "${YELLOW}Running in Docker container${NC}"
else
    echo -e "${YELLOW}Running locally${NC}"
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
fi

# Build command
CMD="python -m src.prediction.predict"
CMD="$CMD --horizon $HORIZON"

if [ -n "$RUN_ID" ]; then
    CMD="$CMD --run-id $RUN_ID"
fi

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

if [ -n "$SCALER" ]; then
    CMD="$CMD --scaler $SCALER"
fi

if [ "$NO_UNCERTAINTY" = true ]; then
    CMD="$CMD --no-uncertainty"
fi

if [ "$NO_SAVE" = true ]; then
    CMD="$CMD --no-save"
fi

echo -e "${GREEN}Generating forecast...${NC}"
echo "Command: $CMD"
echo ""

# Run prediction
$CMD

echo ""
echo -e "${GREEN}Forecast complete!${NC}"
echo -e "${YELLOW}Results saved to: data/outputs/predictions/${NC}"
