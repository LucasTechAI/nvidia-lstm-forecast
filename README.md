# nvidia-lstm-forecast

Deep Learning LSTM model for NVIDIA stock price prediction with MLflow experiment tracking and Docker deployment.

## 🎯 Project Overview

This project implements a complete end-to-end machine learning pipeline for NVIDIA stock price forecasting using:
- **LSTM (Long Short-Term Memory)** neural networks in PyTorch
- **MLflow** for experiment tracking and model management
- **Optuna** for hyperparameter optimization
- **Docker** for reproducible deployment

The system predicts NVIDIA stock prices for the next 30 days using historical data from 2017 onwards.

## 📋 Features

- ✅ Automated ETL pipeline for NVIDIA stock data (Yahoo Finance → SQLite)
- ✅ Configurable LSTM architecture with multiple layers and dropout
- ✅ Comprehensive data preprocessing and sequence generation
- ✅ MLflow integration for experiment tracking and model versioning
- ✅ Bayesian hyperparameter optimization with Optuna
- ✅ 30-day ahead forecasting with visualization
- ✅ Docker containerization for reproducibility
- ✅ Shell scripts for easy execution

## 🏗️ Architecture

```
nvidia-lstm-forecast/
├── src/
│   ├── config.py                    # Centralized configuration
│   ├── data/
│   │   └── preprocessing.py         # Data loading and preprocessing
│   ├── models/
│   │   └── lstm_model.py           # LSTM model implementation
│   ├── training/
│   │   ├── train.py                # Training pipeline
│   │   └── hyperparameter_search.py # Optuna HPO
│   ├── prediction/
│   │   └── predict.py              # Forecasting and visualization
│   ├── etl/                        # Data extraction and loading
│   └── utils/                      # Utility functions
├── scripts/
│   ├── run_training.sh             # Run training
│   ├── run_hpo.sh                  # Run hyperparameter optimization
│   ├── run_prediction.sh           # Generate forecasts
│   └── start_mlflow_ui.sh          # Start MLflow UI
├── data/                           # SQLite database
├── models/                         # Saved models and scalers
├── mlruns/                         # MLflow tracking data
├── outputs/                        # Predictions and plots
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker services
└── requirements.txt                # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional)
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/LucasTechAI/nvidia-lstm-forecast.git
cd nvidia-lstm-forecast
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional)
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running with Docker

1. **Start MLflow UI**
```bash
docker-compose up mlflow
```
Access at http://localhost:5000

2. **Run ETL pipeline**
```bash
docker-compose up etl
```

3. **Train model**
```bash
docker-compose --profile training up training
```

### Running Locally

1. **Run ETL to fetch data**
```bash
python setup/run_etl_nvidia.py
```

2. **Train the model**
```bash
bash scripts/run_training.sh
```

3. **Run hyperparameter optimization** (optional)
```bash
bash scripts/run_hpo.sh 20  # 20 trials
```

4. **Generate predictions**
```bash
# Get the run_id from MLflow UI or logs
bash scripts/run_prediction.sh <mlflow_run_id>
```

5. **Start MLflow UI**
```bash
bash scripts/start_mlflow_ui.sh
```

## ⚙️ Configuration

Key configuration parameters in `src/config.py`:

### Data Parameters
- `DATA_START_YEAR`: 2017 (filter data from this year)
- `TRAIN_SPLIT`: 0.7 (70% for training)
- `VAL_SPLIT`: 0.15 (15% for validation)
- `TEST_SPLIT`: 0.15 (15% for testing)
- `TARGET_COLUMN`: "Close" (prediction target)

### LSTM Architecture
- `SEQUENCE_LENGTH`: 60 (lookback window)
- `HIDDEN_SIZE`: 128 (hidden units)
- `NUM_LAYERS`: 2 (stacked LSTM layers)
- `DROPOUT`: 0.2 (dropout rate)
- `BIDIRECTIONAL`: False (use bidirectional LSTM)

### Training Parameters
- `BATCH_SIZE`: 32
- `EPOCHS`: 100
- `LEARNING_RATE`: 0.001
- `OPTIMIZER`: "Adam"
- `EARLY_STOPPING_PATIENCE`: 10

### Prediction
- `FORECAST_HORIZON`: 30 (days ahead)

## 📊 MLflow Tracking

All experiments are automatically tracked in MLflow:

- **Parameters**: Model architecture, hyperparameters
- **Metrics**: Loss, RMSE, MAE, MAPE (per epoch)
- **Artifacts**: Trained models, scalers, plots
- **Models**: Versioned PyTorch models

Access the MLflow UI at `http://localhost:5000` to:
- Compare experiments
- Visualize training curves
- Download models and artifacts
- Track parameter importance

## 🔬 Hyperparameter Optimization

The system uses Optuna for Bayesian optimization with the following search space:

- `num_layers`: [1, 2, 3, 4]
- `hidden_size`: [32, 64, 128, 256]
- `learning_rate`: [1e-5, 1e-2] (log scale)
- `dropout`: [0.1, 0.5]
- `sequence_length`: [30, 60, 90, 120]
- `batch_size`: [16, 32, 64, 128]

**Objective**: Minimize validation RMSE

Run HPO with:
```bash
bash scripts/run_hpo.sh <n_trials>
```

## 📈 Model Performance

The model is evaluated using:
- **RMSE** (Root Mean Square Error) - primary metric
- **MAE** (Mean Absolute Error) - robust to outliers
- **MAPE** (Mean Absolute Percentage Error) - relative error

Training includes:
- Early stopping to prevent overfitting
- Gradient clipping for stability
- Learning rate scheduling (optional)

## 🐳 Docker Deployment

### Services

1. **mlflow**: MLflow tracking server (port 5000)
2. **etl**: Data extraction and loading
3. **training**: Model training service
4. **dev**: Development environment

### Commands

```bash
# Start MLflow UI
docker-compose up mlflow

# Run ETL
docker-compose up etl

# Train model
docker-compose --profile training up training

# Development mode
docker-compose --profile dev run dev bash
```

## 📝 Usage Examples

### Training a Model

```python
from src.config import settings
from src.data.preprocessing import load_data_from_db, normalize_features, create_sequences
from src.models.lstm_model import create_model
from src.training.train import train_model

# Load and preprocess data
df = load_data_from_db(settings.database_path, start_year=2017)
normalized_data, scaler = normalize_features(df, ['Close'])
X, y = create_sequences(normalized_data, sequence_length=60)

# Create and train model
model = create_model(input_size=1, hidden_size=128, num_layers=2)
trained_model, history = train_model(model, (X_train, y_train), (X_val, y_val), config)
```

### Generating Predictions

```python
from src.prediction.predict import load_best_model, generate_forecast

# Load model from MLflow
model = load_best_model(mlflow_run_id='abc123')

# Generate 30-day forecast
forecast = generate_forecast(model, last_sequence, horizon=30)
```

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

For integration tests:
```bash
pytest tests/ -m integration
```

## 📚 Project Structure Details

### Data Pipeline
1. **ETL**: Fetch NVIDIA data from Yahoo Finance
2. **Storage**: Store in SQLite database
3. **Preprocessing**: Normalize and create sequences
4. **Splitting**: Temporal train/val/test split

### Model Pipeline
1. **Architecture**: Stacked LSTM with dropout
2. **Training**: Adam optimizer, MSE loss
3. **Validation**: Early stopping on val_loss
4. **Logging**: All metrics to MLflow

### Prediction Pipeline
1. **Load**: Best model from MLflow
2. **Forecast**: Iterative 30-day prediction
3. **Inverse**: Transform back to original scale
4. **Visualize**: Plot historical vs forecast

## 🛠️ Development

### Adding New Features

1. Update configuration in `src/config.py`
2. Modify preprocessing in `src/data/preprocessing.py`
3. Adjust model architecture in `src/models/lstm_model.py`
4. Update training pipeline in `src/training/train.py`

### Code Style

- Black for formatting: `black src/`
- Type hints for function signatures
- Docstrings for all public functions

## 🔒 Security

- ✅ All dependencies updated to latest secure versions
- ✅ MLflow ≥3.5.0 (fixes DNS rebinding, RCE, and deserialization vulnerabilities)
- ✅ PyTorch ≥2.6.0 (fixes RCE and memory corruption vulnerabilities)
- ✅ CodeQL analysis: 0 vulnerabilities in application code
- ✅ See [SECURITY.md](SECURITY.md) for detailed security advisory

**Security Best Practices:**
- Never expose MLflow server directly to the internet
- Only load models from trusted sources
- Use `torch.load(..., weights_only=True)` when possible
- Run in isolated Docker containers
- Regularly update dependencies

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

- **Author**: LucasTechAI
- **Email**: lucas.mendestech@gmail.com
- **GitHub**: [LucasTechAI](https://github.com/LucasTechAI)

## 🙏 Acknowledgments

- Yahoo Finance for stock data
- PyTorch team for the deep learning framework
- MLflow team for experiment tracking
- Optuna team for hyperparameter optimization
