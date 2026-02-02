#!/bin/bash
# =============================================================================
# NVIDIA LSTM Forecast - Training Script
# =============================================================================
# Run model training with default or custom parameters
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  NVIDIA LSTM Stock Forecast - Training${NC}"
echo -e "${GREEN}========================================${NC}"

# Default parameters
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-60}
HIDDEN_SIZE=${HIDDEN_SIZE:-128}
NUM_LAYERS=${NUM_LAYERS:-2}
RUN_NAME=${RUN_NAME:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --sequence-length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --hidden-size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs          Number of training epochs (default: 100)"
            echo "  --batch-size      Batch size (default: 32)"
            echo "  --learning-rate   Learning rate (default: 0.001)"
            echo "  --sequence-length Sequence length (default: 60)"
            echo "  --hidden-size     LSTM hidden size (default: 128)"
            echo "  --num-layers      Number of LSTM layers (default: 2)"
            echo "  --run-name        MLflow run name (default: auto-generated)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Training Parameters:${NC}"
echo "  Epochs:          $EPOCHS"
echo "  Batch Size:      $BATCH_SIZE"
echo "  Learning Rate:   $LEARNING_RATE"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Hidden Size:     $HIDDEN_SIZE"
echo "  Num Layers:      $NUM_LAYERS"
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo -e "${YELLOW}Running in Docker container${NC}"
else
    echo -e "${YELLOW}Running locally${NC}"
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
fi

# Build command
CMD="python -m src.training.train"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --sequence-length $SEQUENCE_LENGTH"
CMD="$CMD --hidden-size $HIDDEN_SIZE"
CMD="$CMD --num-layers $NUM_LAYERS"

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run-name $RUN_NAME"
fi

echo -e "${GREEN}Starting training...${NC}"
echo "Command: $CMD"
echo ""

# Run training
$CMD

echo ""
echo -e "${GREEN}Training complete!${NC}"
echo -e "${YELLOW}View results in MLflow UI: http://localhost:5000${NC}"
