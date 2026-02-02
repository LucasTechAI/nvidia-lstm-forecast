#!/bin/bash
# =============================================================================
# NVIDIA LSTM Forecast - Hyperparameter Optimization Script
# =============================================================================
# Run Optuna hyperparameter search with MLflow tracking
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  NVIDIA LSTM - Hyperparameter Optimization${NC}"
echo -e "${GREEN}================================================${NC}"

# Default parameters
N_TRIALS=${N_TRIALS:-50}
TIMEOUT=${TIMEOUT:-""}
RUN_NAME=${RUN_NAME:-""}
TRAIN_BEST=${TRAIN_BEST:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --train-best)
            TRAIN_BEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-trials    Number of Optuna trials (default: 50)"
            echo "  --timeout     Timeout in seconds (default: none)"
            echo "  --run-name    MLflow run name (default: auto-generated)"
            echo "  --train-best  Train with best params after HPO"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}HPO Parameters:${NC}"
echo "  Number of Trials: $N_TRIALS"
echo "  Timeout:          ${TIMEOUT:-none}"
echo "  Train Best:       $TRAIN_BEST"
echo ""

echo -e "${YELLOW}Search Space:${NC}"
echo "  hidden_size:     [32, 64, 128, 256]"
echo "  num_layers:      [1, 2, 3, 4]"
echo "  learning_rate:   [1e-5, 1e-2] (log scale)"
echo "  dropout:         [0.1, 0.5]"
echo "  sequence_length: [30, 60, 90, 120]"
echo "  batch_size:      [16, 32, 64, 128]"
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
CMD="python -m src.training.hyperparameter_search"
CMD="$CMD --n-trials $N_TRIALS"

if [ -n "$TIMEOUT" ]; then
    CMD="$CMD --timeout $TIMEOUT"
fi

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run-name $RUN_NAME"
fi

if [ "$TRAIN_BEST" = true ]; then
    CMD="$CMD --train-best"
fi

echo -e "${GREEN}Starting hyperparameter optimization...${NC}"
echo "Command: $CMD"
echo ""

# Run HPO
$CMD

echo ""
echo -e "${GREEN}HPO complete!${NC}"
echo -e "${YELLOW}View results in MLflow UI: http://localhost:5000${NC}"
echo -e "${YELLOW}Optuna study saved to: data/models/optuna.db${NC}"
