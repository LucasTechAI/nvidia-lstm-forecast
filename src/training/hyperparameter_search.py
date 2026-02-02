"""
Hyperparameter Optimization for NVIDIA LSTM Model.

This module implements Bayesian hyperparameter optimization using Optuna,
with MLflow integration for experiment tracking.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import mlflow
import mlflow.pytorch
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import settings, HPOConfig, TrainingConfig, LSTMConfig
from src.models.lstm_model import NvidiaLSTM
from src.etl.preprocessing import prepare_data_pipeline, create_sequences, normalize_features, load_data_from_db, train_val_test_split, create_data_loaders
from src.training.train import train_epoch, validate_epoch, get_optimizer, get_loss_function, EarlyStopping

# Configure logging
logger = logging.getLogger(__name__)


def create_objective(
    df,
    data_config,
    training_config: TrainingConfig,
    hpo_config: HPOConfig,
    device: torch.device,
    parent_run_id: str
):
    """
    Create an Optuna objective function.
    
    This factory function creates the objective to be optimized, with access
    to shared data and configuration.
    
    Args:
        df: DataFrame with stock data
        data_config: Data configuration
        training_config: Training configuration
        hpo_config: HPO configuration
        device: Device to train on
        parent_run_id: Parent MLflow run ID
        
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function for a single trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation RMSE (to minimize)
        """
        # Sample hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', hpo_config.hidden_size_choices)
        num_layers = trial.suggest_int('num_layers', hpo_config.num_layers_range[0], hpo_config.num_layers_range[1])
        learning_rate = trial.suggest_float('learning_rate', hpo_config.learning_rate_range[0], hpo_config.learning_rate_range[1], log=True)
        dropout = trial.suggest_float('dropout', hpo_config.dropout_range[0], hpo_config.dropout_range[1])
        sequence_length = trial.suggest_categorical('sequence_length', hpo_config.sequence_length_choices)
        batch_size = trial.suggest_categorical('batch_size', hpo_config.batch_size_choices)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial.number}")
        logger.info(f"Hidden size: {hidden_size}, Layers: {num_layers}, LR: {learning_rate:.2e}")
        logger.info(f"Dropout: {dropout:.2f}, Seq len: {sequence_length}, Batch: {batch_size}")
        logger.info(f"{'='*60}")
        
        try:
            # Prepare data with this sequence length
            feature_columns = data_config.feature_columns or [data_config.target_column]
            normalized_data, scaler = normalize_features(
                df,
                feature_columns=feature_columns,
                scaler_type=data_config.scaler_type
            )
            
            X, y = create_sequences(normalized_data, sequence_length)
            
            splits = train_val_test_split(
                X, y,
                train_ratio=data_config.train_split,
                val_ratio=data_config.val_split,
                test_ratio=data_config.test_split
            )
            
            dataloaders = create_data_loaders(splits, batch_size=batch_size)
            
            # Create model
            model = NvidiaLSTM(
                input_size=len(feature_columns),
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1,
                dropout=dropout,
                bidirectional=False
            ).to(device)
            
            # Training setup
            optimizer = get_optimizer(model, training_config)
            optimizer.param_groups[0]['lr'] = learning_rate
            criterion = get_loss_function(training_config)
            early_stopping = EarlyStopping(patience=10, min_delta=1e-6)
            
            # Start nested MLflow run
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                # Log hyperparameters
                mlflow.log_params({
                    'trial_number': trial.number,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'learning_rate': learning_rate,
                    'dropout': dropout,
                    'sequence_length': sequence_length,
                    'batch_size': batch_size
                })
                
                best_val_rmse = float('inf')
                
                # Training loop (reduced epochs for HPO)
                hpo_epochs = min(training_config.epochs, 50)
                
                for epoch in range(1, hpo_epochs + 1):
                    # Train
                    train_metrics = train_epoch(
                        model, dataloaders['train'], optimizer, criterion, device,
                        gradient_clip=training_config.gradient_clip_value
                    )
                    
                    # Validate
                    val_metrics = validate_epoch(
                        model, dataloaders['val'], criterion, device
                    )
                    
                    # Log metrics
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'train_rmse': train_metrics['rmse'],
                        'train_mae': train_metrics['mae'],
                        'train_mape': train_metrics['mape'],
                        'train_r2': train_metrics['r2'],
                        'train_mase': train_metrics['mase'],
                        'train_sharpe': train_metrics['sharpe_ratio'],
                        'val_loss': val_metrics['loss'],
                        'val_rmse': val_metrics['rmse'],
                        'val_mae': val_metrics['mae'],
                        'val_mape': val_metrics['mape'],
                        'val_r2': val_metrics['r2'],
                        'val_correlation': val_metrics['correlation'],
                        'val_directional_accuracy': val_metrics['directional_accuracy'],
                        'val_mase': val_metrics['mase'],
                        'val_theils_u': val_metrics['theils_u'],
                        'val_sharpe': val_metrics['sharpe_ratio'],
                        'val_win_rate': val_metrics['win_rate']
                    }, step=epoch)
                    
                    if val_metrics['rmse'] < best_val_rmse:
                        best_val_rmse = val_metrics['rmse']
                    
                    # Report to Optuna for pruning
                    trial.report(val_metrics['rmse'], epoch)
                    
                    if trial.should_prune():
                        logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                        mlflow.log_metric('pruned', 1)
                        raise optuna.TrialPruned()
                    
                    # Early stopping
                    if early_stopping(val_metrics['loss'], model, epoch):
                        break
                
                # Final evaluation on test set
                test_metrics = validate_epoch(model, dataloaders['test'], criterion, device)
                
                mlflow.log_metrics({
                    'best_val_rmse': best_val_rmse,
                    'test_rmse': test_metrics['rmse'],
                    'test_loss': test_metrics['loss'],
                    'test_mae': test_metrics['mae'],
                    'test_mape': test_metrics['mape'],
                    'test_r2': test_metrics['r2'],
                    'test_correlation': test_metrics['correlation'],
                    'test_directional_accuracy': test_metrics['directional_accuracy'],
                    'test_max_error': test_metrics['max_error'],
                    # Financial/Trading metrics
                    'test_mase': test_metrics['mase'],
                    'test_theils_u': test_metrics['theils_u'],
                    'test_win_rate': test_metrics['win_rate'],
                    'test_profit_factor': test_metrics['profit_factor'],
                    'test_sharpe': test_metrics['sharpe_ratio'],
                    'test_max_drawdown': test_metrics['max_drawdown'],
                    'pruned': 0
                })
                
                logger.info(f"Trial {trial.number} complete - Val RMSE: {best_val_rmse:.6f}, Test RMSE: {test_metrics['rmse']:.6f}")
                
                return best_val_rmse
                
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
    
    return objective


def get_sampler(sampler_name: str) -> optuna.samplers.BaseSampler:
    """Get Optuna sampler by name."""
    samplers = {
        'TPE': TPESampler(seed=42),
        'CMA-ES': CmaEsSampler(seed=42),
        'Random': RandomSampler(seed=42)
    }
    return samplers.get(sampler_name, TPESampler(seed=42))


def get_pruner(pruner_name: str) -> Optional[optuna.pruners.BasePruner]:
    """Get Optuna pruner by name."""
    if pruner_name == 'None':
        return None
    pruners = {
        'MedianPruner': MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        'HyperbandPruner': HyperbandPruner(min_resource=5, max_resource=50)
    }
    return pruners.get(pruner_name, MedianPruner())


def plot_optimization_history(study: optuna.Study, save_path: Path) -> plt.Figure:
    """Plot optimization history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Optimization history
    ax1 = axes[0, 0]
    trials = [t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    ax1.plot(trials, values, 'b.-', alpha=0.7)
    ax1.axhline(y=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.6f}')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Validation RMSE')
    ax1.set_title('Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best value over time
    ax2 = axes[0, 1]
    best_values = []
    current_best = float('inf')
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            current_best = min(current_best, t.value)
            best_values.append(current_best)
    
    ax2.plot(range(len(best_values)), best_values, 'g-', linewidth=2)
    ax2.set_xlabel('Completed Trials')
    ax2.set_ylabel('Best Validation RMSE')
    ax2.set_title('Best Value Progress')
    ax2.grid(True, alpha=0.3)
    
    # Parameter importance
    ax3 = axes[1, 0]
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        values = list(importances.values())
        
        ax3.barh(params, values, color='steelblue')
        ax3.set_xlabel('Importance')
        ax3.set_title('Hyperparameter Importance')
    except Exception as e:
        ax3.text(0.5, 0.5, f'Importance not available:\n{e}', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Distribution of results
    ax4 = axes[1, 1]
    completed_values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ax4.hist(completed_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(x=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.6f}')
    ax4.set_xlabel('Validation RMSE')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Trial Results')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Optimization history saved to {save_path}")
    
    return fig


def plot_parallel_coordinate(study: optuna.Study, save_path: Path) -> Optional[plt.Figure]:
    """Plot parallel coordinate visualization."""
    try:
        ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        # Get the figure from the axes (Optuna returns Axes, not Figure)
        if hasattr(ax, 'figure'):
            fig = ax.figure
        else:
            fig = plt.gcf()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Parallel coordinate plot saved to {save_path}")
        return fig
    except Exception as e:
        logger.warning(f"Could not create parallel coordinate plot: {e}")
        return None


def run_hyperparameter_search(
    hpo_config: Optional[HPOConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization.
    
    Args:
        hpo_config: HPO configuration
        training_config: Training configuration
        run_name: Name for the MLflow parent run
        
    Returns:
        Dictionary with optimization results
    """
    if hpo_config is None:
        hpo_config = settings.hpo
    if training_config is None:
        training_config = settings.training
    
    # Set random seeds
    torch.manual_seed(training_config.random_seed)
    np.random.seed(training_config.random_seed)
    
    # Device setup
    device = torch.device(settings.get_device())
    logger.info(f"Running HPO on device: {device}")
    
    # Load data once
    logger.info("Loading data for HPO...")
    df = load_data_from_db(start_year=settings.data.start_year)
    
    # Setup MLflow
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    # Use study name as experiment name for better organization
    experiment_name = hpo_config.study_name if hpo_config.study_name != settings.hpo.study_name else settings.mlflow.experiment_name
    mlflow.set_experiment(experiment_name)
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"hpo_{timestamp}"
    
    # Create Optuna study
    sampler = get_sampler(hpo_config.sampler)
    pruner = get_pruner(hpo_config.pruner)
    
    study = optuna.create_study(
        study_name=hpo_config.study_name,
        direction=hpo_config.direction,
        sampler=sampler,
        pruner=pruner,
        storage=hpo_config.storage,
        load_if_exists=True
    )
    
    logger.info(f"Created/loaded study: {hpo_config.study_name}")
    logger.info(f"Storage: {hpo_config.storage}")
    logger.info(f"Number of trials: {hpo_config.n_trials}")
    
    # Start parent MLflow run
    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        
        # Log HPO configuration
        mlflow.log_params({
            'hpo_n_trials': hpo_config.n_trials,
            'hpo_sampler': hpo_config.sampler,
            'hpo_pruner': hpo_config.pruner,
            'hpo_metric': hpo_config.metric,
            'search_space_hidden_size': str(hpo_config.hidden_size_choices),
            'search_space_num_layers': str(hpo_config.num_layers_range),
            'search_space_learning_rate': str(hpo_config.learning_rate_range),
            'search_space_dropout': str(hpo_config.dropout_range),
            'search_space_sequence_length': str(hpo_config.sequence_length_choices),
            'search_space_batch_size': str(hpo_config.batch_size_choices)
        })
        
        # Create objective
        objective = create_objective(
            df=df,
            data_config=settings.data,
            training_config=training_config,
            hpo_config=hpo_config,
            device=device,
            parent_run_id=parent_run_id
        )
        
        # Run optimization
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=hpo_config.n_trials,
            timeout=hpo_config.timeout,
            n_jobs=hpo_config.n_jobs,
            show_progress_bar=True
        )
        
        total_time = time.time() - start_time
        
        # Log results
        logger.info("=" * 60)
        logger.info("Hyperparameter Optimization Complete!")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Number of trials: {len(study.trials)}")
        logger.info(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        logger.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        logger.info(f"Best value: {study.best_value:.6f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Log to MLflow
        mlflow.log_metrics({
            'best_val_rmse': study.best_value,
            'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'total_hpo_time': total_time
        })
        
        for param, value in study.best_params.items():
            mlflow.log_param(f'best_{param}', value)
        
        # Save artifacts
        artifact_dir = settings.outputs_dir / 'hpo' / parent_run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save study
        study_path = artifact_dir / 'study.pkl'
        joblib.dump(study, study_path)
        mlflow.log_artifact(str(study_path))
        
        # Plot optimization history
        history_fig = plot_optimization_history(study, artifact_dir / 'optimization_history.png')
        mlflow.log_artifact(str(artifact_dir / 'optimization_history.png'))
        plt.close(history_fig)
        
        # Plot parallel coordinates
        parallel_fig = plot_parallel_coordinate(study, artifact_dir / 'parallel_coordinate.png')
        if parallel_fig:
            mlflow.log_artifact(str(artifact_dir / 'parallel_coordinate.png'))
            plt.close(parallel_fig)
        
        # Save best parameters as JSON
        import json
        best_params_path = artifact_dir / 'best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        mlflow.log_artifact(str(best_params_path))
        
        # Generate importance report
        try:
            importances = optuna.importance.get_param_importances(study)
            importance_path = artifact_dir / 'param_importances.json'
            with open(importance_path, 'w') as f:
                json.dump(importances, f, indent=2)
            mlflow.log_artifact(str(importance_path))
        except Exception as e:
            logger.warning(f"Could not compute parameter importances: {e}")
        
        results = {
            'run_id': parent_run_id,
            'study': study,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'total_time': total_time
        }
        
        return results


def train_with_best_params(
    study: optuna.Study,
    full_training: bool = True
) -> Dict[str, Any]:
    """
    Train a model with the best hyperparameters found.
    
    Args:
        study: Completed Optuna study
        full_training: Whether to train for full epochs
        
    Returns:
        Training results dictionary
    """
    from src.training.train import train_model
    
    best_params = study.best_params
    
    # Update configurations with best params
    lstm_config = LSTMConfig(
        input_size=1,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        output_size=1,
        bidirectional=False,
        sequence_length=best_params['sequence_length']
    )
    
    training_config = settings.training
    training_config.batch_size = best_params['batch_size']
    training_config.learning_rate = best_params['learning_rate']
    
    if full_training:
        training_config.epochs = 100
    
    # Create model
    from src.models.lstm_model import create_model
    model = create_model(config=lstm_config)
    
    # Train
    results = train_model(
        model=model,
        training_config=training_config,
        sequence_length=best_params['sequence_length'],
        run_name='best_model_training'
    )
    
    return results


def main():
    """Main entry point for HPO."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=None, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--run-name', type=str, default=None, help='MLflow run name')
    parser.add_argument('--study-name', type=str, default=None, help='Optuna study name (new name = new experiment)')
    parser.add_argument('--train-best', action='store_true', help='Train with best params after HPO')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format
    )
    
    # Override config
    hpo_config = settings.hpo
    if args.n_trials:
        hpo_config.n_trials = args.n_trials
    if args.timeout:
        hpo_config.timeout = args.timeout
    if args.study_name:
        hpo_config.study_name = args.study_name
        # Also update storage path for new study
        hpo_config.storage = f"sqlite:///./data/outputs/optuna_{args.study_name}.db"
    
    # Run HPO
    results = run_hyperparameter_search(
        hpo_config=hpo_config,
        run_name=args.run_name
    )
    
    print(f"\nHPO Complete!")
    print(f"Run ID: {results['run_id']}")
    print(f"Best Val RMSE: {results['best_value']:.6f}")
    print(f"Best Params: {results['best_params']}")
    
    # Optionally train with best params
    if args.train_best:
        print("\nTraining with best parameters...")
        train_results = train_with_best_params(results['study'])
        print(f"Final Test RMSE: {train_results['test_metrics']['rmse']:.6f}")


if __name__ == '__main__':
    main()
