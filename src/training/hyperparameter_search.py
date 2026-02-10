"""
Hyperparameter optimization using Optuna.

This module performs Bayesian optimization to find the best
hyperparameters for the LSTM model.
"""

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple
import pickle

from src.models.lstm_model import create_model
from src.training.train import train_model

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.Trial,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    input_size: int,
    output_size: int,
    device: torch.device,
    mlflow_tracking_uri: str,
    experiment_name: str
) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        train_data: Training data (X, y)
        val_data: Validation data (X, y)
        input_size: Number of input features
        output_size: Number of output features
        device: Device to train on
        mlflow_tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        
    Returns:
        Validation RMSE to minimize
    """
    # Suggest hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 4)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    sequence_length = trial.suggest_categorical('sequence_length', [30, 60, 90, 120])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Adjust training data sequence length if needed
    # For simplicity in this implementation, we'll use the data as-is
    # In a full implementation, you'd recreate sequences with the suggested length
    
    # Create model
    model = create_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
        output_size=output_size
    )
    model = model.to(device)
    
    # Training config
    config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': 50,  # Reduced for HPO
        'early_stopping_patience': 5,
        'optimizer': 'Adam'
    }
    
    # Setup MLflow for this trial
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Log hyperparameters
        mlflow.log_params({
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'sequence_length': sequence_length,
            'batch_size': batch_size
        })
        
        # Train model
        try:
            trained_model, history = train_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                config=config,
                device=device,
                mlflow_tracking=True
            )
            
            # Get best validation RMSE
            best_val_rmse = min(history['val_rmse'])
            
            # Log final metric
            mlflow.log_metric('best_val_rmse', best_val_rmse)
            
            logger.info(f"Trial {trial.number} - Val RMSE: {best_val_rmse:.6f}")
            
            return best_val_rmse
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}", exc_info=True)
            # Report failed trial to Optuna
            # This allows Optuna to continue with other trials
            raise optuna.TrialPruned(f"Training failed: {str(e)}")


def run_hyperparameter_search(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    input_size: int,
    output_size: int,
    n_trials: int = 50,
    device: torch.device = None,
    mlflow_tracking_uri: str = './mlruns',
    experiment_name: str = 'nvidia-lstm-hpo',
    study_name: str = 'nvidia_lstm_study',
    storage: str = None
) -> Tuple[optuna.Study, Dict]:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        train_data: Training data (X, y)
        val_data: Validation data (X, y)
        input_size: Number of input features
        output_size: Number of output features
        n_trials: Number of trials to run
        device: Device to train on
        mlflow_tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        study_name: Optuna study name
        storage: Optuna storage (e.g., 'sqlite:///optuna.db')
        
    Returns:
        Tuple of (study, best_params)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Starting hyperparameter search with {n_trials} trials")
    logger.info(f"Device: {device}")
    
    # Create or load study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize'
        )
    
    # Setup MLflow callback
    mlflc = MLflowCallback(
        tracking_uri=mlflow_tracking_uri,
        metric_name='val_rmse'
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial=trial,
            train_data=train_data,
            val_data=val_data,
            input_size=input_size,
            output_size=output_size,
            device=device,
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name
        ),
        n_trials=n_trials,
        callbacks=[mlflc]
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best validation RMSE: {best_value:.6f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="hpo_summary"):
        mlflow.log_params(best_params)
        mlflow.log_metric('best_val_rmse', best_value)
        
        # Log parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            for param, imp in importance.items():
                mlflow.log_metric(f'importance_{param}', imp)
            
            logger.info("Parameter importance:")
            for param, imp in importance.items():
                logger.info(f"  {param}: {imp:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {str(e)}")
    
    return study, best_params


def save_study(study: optuna.Study, save_path: str) -> None:
    """
    Save Optuna study to disk.
    
    Args:
        study: Optuna study object
        save_path: Path to save the study
    """
    with open(save_path, 'wb') as f:
        pickle.dump(study, f)
    logger.info(f"Saved study to {save_path}")


def load_study(load_path: str) -> optuna.Study:
    """
    Load Optuna study from disk.
    
    Args:
        load_path: Path to load the study from
        
    Returns:
        Loaded Optuna study
    """
    with open(load_path, 'rb') as f:
        study = pickle.load(f)
    logger.info(f"Loaded study from {load_path}")
    return study


def plot_optimization_history(study: optuna.Study, save_path: str = None) -> None:
    """
    Plot optimization history.
    
    Args:
        study: Optuna study object
        save_path: Optional path to save the plot
    """
    try:
        from optuna.visualization import plot_optimization_history
        fig = plot_optimization_history(study)
        
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Saved optimization history plot to {save_path}")
        
        fig.show()
    except Exception as e:
        logger.warning(f"Could not plot optimization history: {str(e)}")


def plot_param_importances(study: optuna.Study, save_path: str = None) -> None:
    """
    Plot parameter importances.
    
    Args:
        study: Optuna study object
        save_path: Optional path to save the plot
    """
    try:
        from optuna.visualization import plot_param_importances
        fig = plot_param_importances(study)
        
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Saved parameter importance plot to {save_path}")
        
        fig.show()
    except Exception as e:
        logger.warning(f"Could not plot parameter importances: {str(e)}")
