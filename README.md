# NVIDIA Stock LSTM Forecast

<div align="center">

![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Deep Learning LSTM model for NVIDIA (NVDA) stock price prediction**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [API](#-rest-api) • [Dashboard](#-dashboard) • [Docker](#-docker)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Training](#training)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Prediction](#prediction)
- [REST API](#-rest-api)
- [Streamlit Dashboard](#-dashboard)
- [Docker](#-docker)
- [MLflow](#-mlflow)
- [Configuration](#%EF%B8%8F-configuration)
- [Testing](#-testing)
- [License](#-license)

---

## 🎯 Overview

This project implements a complete end-to-end machine learning pipeline for predicting NVIDIA stock prices using **Long Short-Term Memory (LSTM)** neural networks. The system includes:

- **ETL Pipeline**: Automated data extraction and loading from multiple sources
- **LSTM Model**: Deep learning model optimized for time series forecasting
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **MLflow Integration**: Complete experiment tracking and model registry
- **REST API**: FastAPI-powered endpoints for inference and training
- **Interactive Dashboard**: Streamlit-based visualization and forecasting
- **Docker Support**: Containerized deployment for all services

### Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.9064 |
| MAPE | 4.95% |
| Correlation | 0.9728 |
| Directional Accuracy | 54.2% |

---

## ✨ Features

### Core Features
- 🔮 **Multi-horizon Forecasting**: Predict 7, 30, 60, or 90 days ahead
- 🧠 **LSTM Neural Network**: 2-layer stacked LSTM with dropout regularization
- 📊 **Interactive Dashboard**: Real-time predictions and model metrics visualization
- 🚀 **REST API**: Production-ready API for inference and training
- 📈 **MLflow Tracking**: Complete experiment tracking with artifacts

### Technical Features
- ⚡ **Hyperparameter Optimization**: Optuna-powered automated tuning
- 🐳 **Docker Containerization**: Multi-service Docker Compose setup
- 🧪 **Comprehensive Testing**: Unit tests with pytest
- 📁 **ETL Pipeline**: Automated data extraction and processing
- 🔄 **Model Versioning**: MLflow model registry integration

---

## 📁 Project Structure

```
nvidia-lstm-forecast/
├── src/
│   ├── config.py                # Centralized configuration
│   ├── api/                     # FastAPI REST API
│   │   ├── main.py              # API application entry point
│   │   ├── dependencies.py      # Dependency injection
│   │   ├── schemas.py           # Pydantic schemas
│   │   └── routers/
│   │       ├── health.py        # Health check endpoints
│   │       ├── predict.py       # Prediction endpoints
│   │       ├── train.py         # Training endpoints
│   │       └── data.py          # Data endpoints
│   ├── dashboard/               # Streamlit dashboard
│   │   ├── app.py               # Dashboard entry point
│   │   └── components/
│   │       ├── predictions.py   # Forecast visualization
│   │       ├── metrics.py       # Model metrics display
│   │       ├── model_schema.py  # Architecture visualization
│   │       └── sidebar.py       # Navigation sidebar
│   ├── etl/
│   │   ├── extractor_nvidia.py  # Data extraction
│   │   ├── load_sqlite_nvidia.py # SQLite loading
│   │   └── preprocessing.py     # Data preprocessing
│   ├── models/
│   │   └── lstm_model.py        # LSTM model architecture
│   ├── training/
│   │   ├── train.py             # Training pipeline
│   │   └── hyperparameter_search.py  # Optuna HPO
│   ├── prediction/
│   │   └── predict.py           # Forecasting module
│   └── utils/
│       └── database_manager.py  # Database utilities
├── data/
│   ├── raw/
│   │   └── nvidia_stock.csv     # Raw stock data
│   ├── nvidia_stock.db          # SQLite database
│   ├── models/                  # Trained models
│   │   └── checkpoints/
│   │       ├── best_model.pt    # Best trained model
│   │       └── latest_checkpoint.pt
│   ├── outputs/                 # Training outputs
│   │   ├── artifacts/           # Scalers and artifacts
│   │   └── hpo/                 # HPO results
│   └── mlruns/                  # MLflow tracking data
├── scripts/
│   ├── run_training.sh          # Training script
│   ├── run_hpo.sh               # HPO script
│   ├── run_prediction.sh        # Prediction script
│   ├── run_dashboard.sh         # Dashboard script
│   ├── start_mlflow_ui.sh       # MLflow UI script
│   └── docker_helper.sh         # Docker helper commands
├── notebooks/
│   ├── EDA.ipynb                # Exploratory Data Analysis
│   └── model_metrics_analysis.ipynb  # Model metrics analysis
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── Dockerfile
├── Dockerfile.api
├── docker-compose.yml
├── docker-compose.api.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- pip or conda
- Docker & Docker Compose (optional, for containerized deployment)
- CUDA-compatible GPU (optional, for faster training)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nvidia-lstm-forecast.git
cd nvidia-lstm-forecast
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

### Docker Installation

```bash
# Build images
docker compose build

# Start all services
docker compose up -d
```

---

## ⚡ Quick Start

### 1. Start MLflow UI
```bash
./scripts/start_mlflow_ui.sh
# Visit http://localhost:5000
```

### 2. Train a Model
```bash
python -m src.training.train --epochs 50 --batch-size 32
```

### 3. Run Hyperparameter Optimization
```bash
python -m src.training.hyperparameter_search --n-trials 50
```

### 4. Generate Predictions
```bash
python -m src.prediction.predict --horizon 30
```

### 5. Start the Dashboard
```bash
streamlit run src/dashboard/app.py
# Visit http://localhost:8501
```

### 6. Start the API
```bash
uvicorn api.main:app --reload --port 8000
# Visit http://localhost:8000/docs
```

---

## 📖 Usage

### Training

Train a model with default parameters:
```bash
python -m src.training.train
```

With custom parameters:
```bash
python -m src.training.train \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --sequence-length 60 \
    --hidden-size 128 \
    --num-layers 2
```

### Hyperparameter Optimization

Run HPO with Optuna:
```bash
python -m src.training.hyperparameter_search --n-trials 50
```

With study name for experiment tracking:
```bash
python -m src.training.hyperparameter_search \
    --n-trials 50 \
    --study-name "experiment_v1"
```

### Prediction

Generate 30-day forecast:
```bash
python -m src.prediction.predict --horizon 30
```

Using specific MLflow run:
```bash
python -m src.prediction.predict \
    --run-id <mlflow-run-id> \
    --horizon 30
```

---

## 🔌 REST API

The project includes a production-ready FastAPI REST API.

### Starting the API

```bash
# Development
uvicorn api.main:app --reload --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/health/model` | Model status |
| `POST` | `/predict` | Generate predictions |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/data/historical` | Get historical data |
| `GET` | `/data/latest` | Get latest price |
| `POST` | `/train` | Trigger training |
| `GET` | `/train/status/{job_id}` | Training job status |

### Example Requests

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Generate Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"horizon": 30}'
```

**Get Historical Data:**
```bash
curl "http://localhost:8000/data/historical?days=90"
```

### API Documentation

Interactive API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📊 Dashboard

The Streamlit dashboard provides an interactive interface for:

### Features

- **📊 Stock Predictions**: Multi-horizon forecasting (7, 30, 60, 90 days)
- **📈 Model Metrics**: Performance metrics visualization (R², RMSE, MAE, MAPE)
- **🧠 Model Architecture**: Interactive model tree and architecture diagram
- **📥 Data Export**: Download predictions as CSV

### Starting the Dashboard

```bash
# Using script
./scripts/run_dashboard.sh

# Or directly
streamlit run src/dashboard/app.py --server.port 8501
```

Access the dashboard at: http://localhost:8501

### Dashboard Pages

1. **Stock Predictions**
   - Configure forecast horizon (7, 30, 60, 90 days)
   - Interactive price chart with confidence intervals
   - Historical context visualization
   - Daily changes analysis
   - Export predictions to CSV

2. **Model Metrics**
   - Training overview (epochs, loss, early stopping)
   - Test set performance (R², RMSE, MAE, MAPE)
   - Training curves visualization
   - Hyperparameter importance analysis

3. **Model Architecture**
   - Model tree structure visualization
   - Layer configuration details
   - Parameter distribution analysis
   - Data flow diagram
   - Export model configuration

---

## 🐳 Docker

### Available Services

| Service | Description | Port |
|---------|-------------|------|
| `mlflow` | MLflow tracking server | 5000 |
| `api` | FastAPI REST API | 8000 |
| `dashboard` | Streamlit dashboard | 8501 |
| `training` | Model training service | - |
| `hpo` | Hyperparameter optimization | - |
| `prediction` | Forecast generation | - |
| `etl` | Data extraction pipeline | - |

### Docker Commands

```bash
# Start all services
docker compose up -d

# Start specific service
docker compose up -d mlflow api dashboard

# Run training
docker compose run --rm training

# Run HPO
docker compose run --rm hpo

# View logs
docker compose logs -f api

# Stop all services
docker compose down
```

### Using Docker Helper Script

```bash
# Start MLflow server
./scripts/docker_helper.sh mlflow

# Run training
./scripts/docker_helper.sh train

# Run HPO
./scripts/docker_helper.sh hpo

# Generate predictions
./scripts/docker_helper.sh predict

# Run full pipeline
./scripts/docker_helper.sh full-pipeline

# Stop all services
./scripts/docker_helper.sh stop
```

---

## 📊 MLflow

### Accessing the UI

1. Start the MLflow server:
```bash
./scripts/start_mlflow_ui.sh
```

2. Open http://localhost:5000 in your browser

### Tracked Information

- **Parameters**: Learning rate, batch size, epochs, model architecture
- **Metrics**: Training/validation loss, R², RMSE, MAE, MAPE
- **Artifacts**:
  - Trained model weights (`.pt` files)
  - Scaler (for inverse transformation)
  - Loss curves
  - Prediction plots
  - HPO study results

### Loading a Model from MLflow

```python
import mlflow.pytorch

# Load by run ID
model = mlflow.pytorch.load_model("runs:/<run-id>/model")

# Load from model registry
model = mlflow.pytorch.load_model("models:/nvidia-lstm-model/latest")
```

---

## ⚙️ Configuration

All configuration is centralized in `src/config.py`:

### Data Configuration
```python
DataConfig(
    start_year=2017,          # Only use data from 2017+
    train_split=0.7,          # 70% training
    val_split=0.15,           # 15% validation
    test_split=0.15,          # 15% testing
    target_column='Close',    # Predict closing price
    scaler_type='MinMaxScaler'
)
```

### Model Configuration
```python
LSTMConfig(
    sequence_length=60,       # 60-day lookback window
    hidden_size=128,          # LSTM hidden units
    num_layers=2,             # Stacked LSTM layers
    dropout=0.2,              # Dropout rate
    bidirectional=False       # Unidirectional LSTM
)
```

### Training Configuration
```python
TrainingConfig(
    batch_size=32,
    epochs=100,
    learning_rate=0.001,
    optimizer='Adam',
    early_stopping_patience=10,
    gradient_clip_value=1.0
)
```

### HPO Search Space
```python
HPOConfig(
    n_trials=50,
    hidden_size_choices=[32, 64, 128, 256],
    num_layers_range=(1, 4),
    learning_rate_range=(1e-5, 1e-2),
    dropout_range=(0.1, 0.5),
    sequence_length_choices=[30, 60, 90, 120],
    batch_size_choices=[16, 32, 64, 128]
)
```

---

## 📚 API Reference

### Data Module

```python
from src.etl.preprocessing import (
    load_data_from_db,
    normalize_features,
    create_sequences,
    train_val_test_split,
    prepare_data_pipeline
)

# Load data
df = load_data_from_db(start_year=2017)

# Full pipeline
dataloaders, scaler, df = prepare_data_pipeline(
    sequence_length=60,
    batch_size=32
)
```

### Model Module

```python
from src.models.lstm_model import NvidiaLSTM, create_model

# Create model with defaults
model = create_model()

# Custom configuration
model = NvidiaLSTM(
    input_size=1,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    bidirectional=False
)
```

### Training Module

```python
from src.training.train import train_model

results = train_model(
    sequence_length=60,
    run_name="my_experiment"
)

print(f"Run ID: {results['run_id']}")
print(f"Test RMSE: {results['test_metrics']['rmse']}")
```

### Prediction Module

```python
from src.prediction.predict import run_prediction_pipeline

result = run_prediction_pipeline(
    run_id="<mlflow-run-id>",
    horizon=30,
    with_uncertainty=True
)

# Access results
print(result.to_dataframe())
```

---

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_models/test_lstm_model.py -v
```

---

## 🗺️ Roadmap

- [ ] Add Transformer-based model alternative
- [ ] Implement real-time data streaming
- [ ] Add technical indicators as features
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Kubernetes deployment configuration
- [ ] Model monitoring and alerting

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**This project is for educational purposes only.** Stock price predictions should not be used for actual trading decisions. The model's predictions are based on historical patterns and do not account for market events, news, or other factors that can significantly impact stock prices. Always consult with a financial advisor before making investment decisions.

---

<div align="center">

**Built with ❤️ using PyTorch, FastAPI, Streamlit & MLflow**

</div>
