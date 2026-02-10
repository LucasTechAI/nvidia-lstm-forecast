#!/bin/bash
# =============================================================================
# NVIDIA LSTM Forecast - Docker Compose Helpers
# =============================================================================
# Convenient scripts for Docker Compose operations
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

COMMAND=${1:-"help"}

case $COMMAND in
    # =========================================================================
    # MLflow Server
    # =========================================================================
    mlflow)
        echo -e "${GREEN}Starting MLflow server...${NC}"
        docker compose up -d mlflow
        echo -e "${GREEN}MLflow UI available at: http://localhost:5000${NC}"
        ;;
    
    # =========================================================================
    # Training
    # =========================================================================
    train)
        echo -e "${GREEN}Starting training service...${NC}"
        docker compose --profile training up training
        ;;
    
    # =========================================================================
    # Hyperparameter Optimization
    # =========================================================================
    hpo)
        echo -e "${GREEN}Starting HPO service...${NC}"
        docker compose --profile hpo up hpo
        ;;
    
    # =========================================================================
    # Prediction
    # =========================================================================
    predict)
        echo -e "${GREEN}Starting prediction service...${NC}"
        docker compose --profile prediction up prediction
        ;;
    
    # =========================================================================
    # ETL Pipeline
    # =========================================================================
    etl)
        echo -e "${GREEN}Running ETL pipeline...${NC}"
        docker compose up etl
        ;;
    
    # =========================================================================
    # Development Environment
    # =========================================================================
    dev)
        echo -e "${GREEN}Starting development environment...${NC}"
        docker compose --profile dev run --rm dev
        ;;
    
    # =========================================================================
    # Build
    # =========================================================================
    build)
        echo -e "${GREEN}Building Docker images...${NC}"
        docker compose build
        ;;
    
    # =========================================================================
    # Stop All
    # =========================================================================
    stop)
        echo -e "${YELLOW}Stopping all services...${NC}"
        docker compose down
        ;;
    
    # =========================================================================
    # Logs
    # =========================================================================
    logs)
        SERVICE=${2:-""}
        if [ -n "$SERVICE" ]; then
            docker compose logs -f "$SERVICE"
        else
            docker compose logs -f
        fi
        ;;
    
    # =========================================================================
    # Status
    # =========================================================================
    status)
        echo -e "${GREEN}Service Status:${NC}"
        docker compose ps -a
        ;;
    
    # =========================================================================
    # Clean
    # =========================================================================
    clean)
        echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
        docker compose down -v --remove-orphans
        docker system prune -f
        ;;
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    full-pipeline)
        echo -e "${GREEN}Running full pipeline...${NC}"
        echo ""
        
        # Start MLflow
        echo -e "${YELLOW}1. Starting MLflow server...${NC}"
        docker compose up -d mlflow
        sleep 5
        
        # Run ETL
        echo -e "${YELLOW}2. Running ETL pipeline...${NC}"
        docker compose up etl
        
        # Run HPO
        echo -e "${YELLOW}3. Running hyperparameter optimization...${NC}"
        docker compose --profile hpo up hpo
        
        # Run Prediction
        echo -e "${YELLOW}4. Generating predictions...${NC}"
        docker compose --profile prediction up prediction
        
        echo ""
        echo -e "${GREEN}Pipeline complete!${NC}"
        echo -e "${YELLOW}View results at: http://localhost:5000${NC}"
        ;;
    
    # =========================================================================
    # Help
    # =========================================================================
    help|*)
        echo -e "${GREEN}NVIDIA LSTM Forecast - Docker Helper${NC}"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  mlflow         Start MLflow tracking server"
        echo "  train          Run model training"
        echo "  hpo            Run hyperparameter optimization"
        echo "  predict        Generate forecasts"
        echo "  etl            Run ETL data pipeline"
        echo "  dev            Start development environment"
        echo "  build          Build Docker images"
        echo "  stop           Stop all services"
        echo "  logs [service] View logs (optionally for specific service)"
        echo "  status         Show service status"
        echo "  clean          Clean up Docker resources"
        echo "  full-pipeline  Run complete pipeline (ETL -> HPO -> Predict)"
        echo "  help           Show this help message"
        ;;
esac
