#!/bin/bash
# Start MLflow UI server

set -e

echo "Starting MLflow UI..."

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="./mlruns"

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

echo "MLflow UI is running at http://localhost:5000"
