#!/bin/bash
# =============================================================================
# Run Streamlit Dashboard
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default port
PORT=${1:-8501}

echo -e "${GREEN}Starting NVIDIA Stock Prediction Dashboard...${NC}"
echo -e "${YELLOW}Dashboard will be available at: http://localhost:${PORT}${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Run Streamlit
cd "$PROJECT_ROOT"
streamlit run src/dashboard/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.primaryColor="#76B900" \
    --theme.backgroundColor="#0E1117" \
    --theme.secondaryBackgroundColor="#262730" \
    --theme.textColor="#FAFAFA"
