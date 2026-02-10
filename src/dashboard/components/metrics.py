"""
Model Metrics Component for Dashboard.

Displays model performance metrics and training history.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import torch
import sqlite3
import json

project_root = Path(__file__).resolve().parent.parent.parent.parent


def load_checkpoint_info() -> dict:
    """Load information from the best model checkpoint."""
    checkpoint_path = project_root / "data" / "models" / "checkpoints" / "best_model.pt"
    
    if not checkpoint_path.exists():
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None


def load_mlflow_metrics() -> pd.DataFrame:
    """Load metrics from MLflow database."""
    db_path = project_root / "data" / "mlruns" / "mlflow.db"
    
    if not db_path.exists():
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get the latest run
        runs_query = """
            SELECT run_uuid, start_time, end_time, status, name
            FROM runs
            WHERE status = 'FINISHED'
            ORDER BY start_time DESC
            LIMIT 1
        """
        runs = pd.read_sql_query(runs_query, conn)
        
        if runs.empty:
            conn.close()
            return pd.DataFrame()
        
        latest_run_id = runs.iloc[0]['run_uuid']
        
        # Get metrics for the latest run
        metrics_query = f"""
            SELECT key, value, step, timestamp
            FROM metrics
            WHERE run_uuid = '{latest_run_id}'
            ORDER BY key, step
        """
        metrics_df = pd.read_sql_query(metrics_query, conn)
        conn.close()
        
        return metrics_df
        
    except Exception as e:
        st.error(f"Error loading MLflow metrics: {e}")
        return pd.DataFrame()


def load_hpo_results() -> dict:
    """Load HPO best parameters."""
    hpo_paths = list((project_root / "data" / "outputs" / "hpo").glob("*/best_params.json"))
    
    if not hpo_paths:
        return None
    
    # Get the most recent one
    latest = max(hpo_paths, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest, 'r') as f:
            return json.load(f)
    except:
        return None


def render_metrics_page():
    """Render the model metrics page."""
    
    # Enhanced page header
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-weight: 600;">📈 Model Performance Metrics</h2>
            <p style="color: rgba(250,250,250,0.6); margin-top: 0.5rem;">
                Explore training history, evaluation metrics, and performance analysis of the LSTM model.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading model metrics..."):
        checkpoint = load_checkpoint_info()
        mlflow_metrics = load_mlflow_metrics()
        hpo_params = load_hpo_results()
    
    if checkpoint is None:
        st.markdown("""
            <div style="
                background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
                border: 1px solid rgba(255, 193, 7, 0.3);
                border-radius: 12px;
                padding: 24px;
                text-align: center;
            ">
                <span style="font-size: 2.5rem;">🤖</span>
                <h3 style="margin: 0.5rem 0; color: #FFC107;">No Model Found</h3>
                <p style="color: rgba(250,250,250,0.7); margin: 0;">
                    Please train the model first to view performance metrics.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Training Overview Section
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            🎯 Training Overview
        </p>
    """, unsafe_allow_html=True)
    
    training_info = checkpoint.get('training_info', {})
    model_config = checkpoint.get('model_config', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Best Epoch",
            value=training_info.get('Best Epoch', checkpoint.get('epoch', 'N/A'))
        )
    
    with col2:
        best_loss = training_info.get('Best Val Loss', checkpoint.get('loss', 0))
        st.metric(
            label="Best Validation Loss",
            value=f"{best_loss:.6f}" if isinstance(best_loss, (int, float)) else str(best_loss)
        )
    
    with col3:
        total_epochs = training_info.get('Total Epochs', checkpoint.get('epoch', 'N/A'))
        st.metric(
            label="Total Epochs Trained",
            value=total_epochs
        )
    
    with col4:
        early_stop = training_info.get('Early Stopped', False)
        st.metric(
            label="Early Stopping",
            value="Yes ✅" if early_stop else "No ❌"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Test Metrics Section
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            📊 Test Set Performance
        </p>
    """, unsafe_allow_html=True)
    
    test_results = checkpoint.get('test_results', {})
    
    if test_results:
        # Primary metrics in larger cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            r2 = test_results.get('r2_score', test_results.get('R2 Score', 0))
            quality = "🟢 Excellent" if r2 > 0.9 else ("🟡 Good" if r2 > 0.7 else "🔴 Needs Work")
            st.metric(
                label="R² Score",
                value=f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2),
                delta=quality
            )
        
        with col2:
            rmse = test_results.get('rmse', test_results.get('RMSE', 0))
            st.metric(
                label="RMSE",
                value=f"${rmse:.2f}" if isinstance(rmse, (int, float)) else str(rmse)
            )
        
        with col3:
            mae = test_results.get('mae', test_results.get('MAE', 0))
            st.metric(
                label="MAE",
                value=f"${mae:.2f}" if isinstance(mae, (int, float)) else str(mae)
            )
        
        with col4:
            mape = test_results.get('mape', test_results.get('MAPE', 0))
            st.metric(
                label="MAPE",
                value=f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape)
            )
        
        # Additional metrics in secondary row
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            corr = test_results.get('correlation', test_results.get('Correlation', 0))
            st.metric(
                label="Correlation",
                value=f"{corr:.4f}" if isinstance(corr, (int, float)) else str(corr)
            )
        
        with col2:
            dir_acc = test_results.get('directional_accuracy', test_results.get('Directional Accuracy', 0))
            st.metric(
                label="Directional Accuracy",
                value=f"{dir_acc:.1f}%" if isinstance(dir_acc, (int, float)) else str(dir_acc)
            )
        
        with col3:
            sharpe = test_results.get('sharpe_ratio', test_results.get('Sharpe Ratio', 0))
            st.metric(
                label="Sharpe Ratio",
                value=f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
            )
        
        with col4:
            max_dd = test_results.get('max_drawdown', test_results.get('Max Drawdown', 0))
            st.metric(
                label="Max Drawdown",
                value=f"{max_dd:.1f}%" if isinstance(max_dd, (int, float)) else str(max_dd)
            )
    else:
        st.markdown("""
            <div style="
                background: linear-gradient(145deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.05) 100%);
                border: 1px solid rgba(33, 150, 243, 0.3);
                border-radius: 10px;
                padding: 16px;
            ">
                <span style="color: #2196F3;">ℹ️</span> Test results not available in checkpoint.
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Training History Visualization
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            📉 Training History
        </p>
    """, unsafe_allow_html=True)
    
    if not mlflow_metrics.empty:
        render_training_curves(mlflow_metrics)
    else:
        # Try to get from checkpoint
        if 'train_losses' in checkpoint or 'val_losses' in checkpoint:
            render_training_curves_from_checkpoint(checkpoint)
        else:
            st.markdown("""
                <div style="
                    background: linear-gradient(145deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.05) 100%);
                    border: 1px solid rgba(33, 150, 243, 0.3);
                    border-radius: 10px;
                    padding: 16px;
                ">
                    <span style="color: #2196F3;">ℹ️</span> Training history not available.
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics Interpretation
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            📝 Metrics Interpretation
        </p>
    """, unsafe_allow_html=True)
    
    render_metrics_interpretation(test_results)
    
    # HPO Results
    if hpo_params:
        st.markdown("---")
        st.markdown("""
            <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
                🔬 Best Hyperparameters (from HPO)
            </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            params_df = pd.DataFrame([
                {"Parameter": k, "Value": v}
                for k, v in hpo_params.items()
            ])
            st.dataframe(params_df, width='stretch', hide_index=True)
        
        with col2:
            # Radar chart of normalized parameters
            render_hpo_radar(hpo_params)


def render_training_curves(metrics_df: pd.DataFrame):
    """Render training curves from MLflow metrics."""
    
    # Filter for training metrics
    loss_metrics = metrics_df[metrics_df['key'].str.contains('loss', case=False)]
    
    if loss_metrics.empty:
        st.info("No loss metrics found in MLflow.")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss', 'RMSE', 'MAE', 'R² Score'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    metric_configs = [
        ('loss', 1, 1, '#FF6B6B'),
        ('rmse', 1, 2, '#4ECDC4'),
        ('mae', 2, 1, '#45B7D1'),
        ('r2', 2, 2, '#96CEB4')
    ]
    
    for metric_key, row, col, color in metric_configs:
        # Get train and val metrics
        train_data = metrics_df[
            (metrics_df['key'].str.contains(f'train_{metric_key}', case=False)) |
            (metrics_df['key'].str.contains(f'{metric_key}', case=False) & ~metrics_df['key'].str.contains('val|test', case=False))
        ]
        val_data = metrics_df[metrics_df['key'].str.contains(f'val_{metric_key}', case=False)]
        
        if not train_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=train_data['step'],
                    y=train_data['value'],
                    mode='lines',
                    name=f'Train {metric_key.upper()}',
                    line=dict(color=color),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
        
        if not val_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=val_data['step'],
                    y=val_data['value'],
                    mode='lines',
                    name=f'Val {metric_key.upper()}',
                    line=dict(color=color, dash='dash'),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        title=dict(text='Training Metrics Over Epochs', font=dict(size=18))
    )
    
    st.plotly_chart(fig, width='stretch')


def render_training_curves_from_checkpoint(checkpoint: dict):
    """Render training curves from checkpoint data."""
    
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    if not train_losses and not val_losses:
        st.info("No training history in checkpoint.")
        return
    
    fig = go.Figure()
    
    if train_losses:
        fig.add_trace(go.Scatter(
            y=train_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='#76B900', width=2)
        ))
    
    if val_losses:
        fig.add_trace(go.Scatter(
            y=val_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='#FF6B35', width=2)
        ))
    
    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss (MSE)',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')


def render_metrics_interpretation(test_results: dict):
    """Render interpretation of metrics."""
    
    interpretations = []
    
    if test_results:
        r2 = test_results.get('r2_score', test_results.get('R2 Score', 0))
        if isinstance(r2, (int, float)):
            if r2 > 0.95:
                interpretations.append(("R² Score", "🟢 Excellent", f"The model explains {r2*100:.1f}% of the variance in stock prices."))
            elif r2 > 0.85:
                interpretations.append(("R² Score", "🟡 Good", f"The model explains {r2*100:.1f}% of the variance. Room for improvement."))
            else:
                interpretations.append(("R² Score", "🔴 Needs Improvement", f"Only {r2*100:.1f}% variance explained."))
        
        mape = test_results.get('mape', test_results.get('MAPE', 0))
        if isinstance(mape, (int, float)):
            if mape < 5:
                interpretations.append(("MAPE", "🟢 Excellent", f"Average prediction error is only {mape:.2f}%."))
            elif mape < 10:
                interpretations.append(("MAPE", "🟡 Good", f"Average prediction error of {mape:.2f}% is acceptable."))
            else:
                interpretations.append(("MAPE", "🔴 High Error", f"Average prediction error of {mape:.2f}% is concerning."))
        
        dir_acc = test_results.get('directional_accuracy', test_results.get('Directional Accuracy', 0))
        if isinstance(dir_acc, (int, float)):
            if dir_acc > 55:
                interpretations.append(("Directional Accuracy", "🟢 Good", f"Model correctly predicts direction {dir_acc:.1f}% of the time."))
            elif dir_acc > 50:
                interpretations.append(("Directional Accuracy", "🟡 Marginal", f"Slightly better than random at {dir_acc:.1f}%."))
            else:
                interpretations.append(("Directional Accuracy", "🔴 Poor", f"Only {dir_acc:.1f}% - worse than random guessing."))
    
    if interpretations:
        for metric, status, description in interpretations:
            st.markdown(f"**{metric}** {status}")
            st.markdown(f"_{description}_")
            st.markdown("")
    else:
        st.info("ℹ️ Interpretation requires test metrics.")


def render_hpo_radar(params: dict):
    """Render radar chart of hyperparameters."""
    
    # Normalize parameters for visualization
    param_ranges = {
        'hidden_size': (32, 256),
        'num_layers': (1, 4),
        'learning_rate': (0.0001, 0.01),
        'dropout': (0.0, 0.5),
        'sequence_length': (30, 120),
        'batch_size': (16, 128)
    }
    
    categories = []
    values = []
    
    for param, (min_val, max_val) in param_ranges.items():
        if param in params:
            val = params[param]
            # Normalize to 0-1
            if param == 'learning_rate':
                normalized = (np.log10(val) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val))
            else:
                normalized = (val - min_val) / (max_val - min_val)
            categories.append(param.replace('_', ' ').title())
            values.append(max(0, min(1, normalized)))
    
    if not categories:
        return
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(118, 185, 0, 0.3)',
        line=dict(color='#76B900', width=2),
        name='Best Parameters'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            )
        ),
        template='plotly_dark',
        height=300,
        margin=dict(l=60, r=60, t=40, b=40),
        title=dict(text='Parameter Distribution', font=dict(size=14))
    )
    
    st.plotly_chart(fig, width='stretch')
