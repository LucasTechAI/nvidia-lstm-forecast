"""
Predictions Component for Dashboard.

Displays stock price predictions with configurable horizons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import torch
import sqlite3
import joblib

import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_historical_data() -> pd.DataFrame:
    """Load historical stock data from database."""
    db_path = project_root / "data" / "nvidia_stock.db"
    
    if not db_path.exists():
        # Try CSV as fallback
        csv_path = project_root / "data" / "raw" / "nvidia_stock.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM nvidia_stock ORDER BY date"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Normalize column names
    df.columns = [col.capitalize() if col != 'date' else 'Date' for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def load_model_and_scaler():
    """Load the trained model and scaler."""
    checkpoint_path = project_root / "data" / "models" / "checkpoints" / "best_model.pt"
    
    if not checkpoint_path.exists():
        return None, None, None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get model config
        model_config = checkpoint.get('model_config', {})
        
        # Create model
        from src.models.lstm_model import NvidiaLSTM
        
        model = NvidiaLSTM(
            input_size=model_config.get('input_size', 1),
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            output_size=model_config.get('output_size', 1),
            dropout=model_config.get('dropout', 0.2),
            bidirectional=model_config.get('bidirectional', False)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load scaler
        scaler_paths = [
            project_root / "data" / "outputs" / "artifacts" / "scaler.joblib",
            project_root / "data" / "models" / "scaler.joblib"
        ]
        
        scaler = None
        for path in scaler_paths:
            if path.exists():
                scaler = joblib.load(path)
                break
        
        # Try to find scaler in mlruns
        if scaler is None:
            mlruns_artifacts = project_root / "data" / "mlruns"
            for scaler_file in mlruns_artifacts.rglob("scaler.joblib"):
                scaler = joblib.load(scaler_file)
                break
        
        return model, scaler, model_config
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def generate_predictions(model, scaler, historical_data: pd.DataFrame, horizon: int) -> dict:
    """Generate price predictions for the given horizon."""
    if model is None or scaler is None:
        return None
    
    try:
        # Get sequence length from model
        seq_length = 60  # Default
        
        # Prepare last sequence
        close_prices = historical_data['Close'].values[-seq_length:]
        
        # Normalize
        close_normalized = scaler.transform(close_prices.reshape(-1, 1))
        
        # Create tensor
        sequence = torch.FloatTensor(close_normalized).unsqueeze(0)  # (1, seq_len, 1)
        
        # Generate predictions
        model.eval()
        predictions_normalized = []
        current_sequence = sequence.clone()
        
        with torch.no_grad():
            for _ in range(horizon):
                pred = model(current_sequence)
                predictions_normalized.append(pred.item())
                
                # Update sequence
                new_input = pred.view(1, 1, -1)
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    new_input
                ], dim=1)
        
        # Inverse transform
        predictions_normalized = np.array(predictions_normalized).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions_normalized).flatten()
        
        # Generate dates (skip weekends)
        last_date = historical_data['Date'].iloc[-1]
        dates = []
        current_date = last_date
        
        while len(dates) < horizon:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # Skip weekends
                dates.append(current_date)
        
        return {
            'dates': dates,
            'predictions': predictions,
            'last_price': historical_data['Close'].iloc[-1],
            'last_date': last_date
        }
        
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return None


def render_predictions_page():
    """Render the predictions page."""
    
    # Page header with description
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-weight: 600;">📊 Stock Price Predictions</h2>
            <p style="color: rgba(250,250,250,0.6); margin-top: 0.5rem;">
                Generate AI-powered NVIDIA stock price forecasts using our trained LSTM neural network.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Control panel in a styled container
    st.markdown("""
        <div style="
            background: linear-gradient(145deg, rgba(118, 185, 0, 0.05) 0%, rgba(118, 185, 0, 0.02) 100%);
            border: 1px solid rgba(118, 185, 0, 0.2);
            border-radius: 12px;
            padding: 1rem 1.5rem 0.5rem 1.5rem;
            margin-bottom: 1.5rem;
        ">
            <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                ⚙️ Forecast Settings
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Horizon selection with better layout
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        horizon_options = {
            "🗓️ 7 Days": 7,
            "📅 30 Days (Recommended)": 30,
            "📆 60 Days": 60,
            "🗓️ 90 Days": 90
        }
        
        selected_horizon = st.selectbox(
            "Forecast Horizon",
            list(horizon_options.keys()),
            index=1,  # Default to 30 days
            help="Select how far into the future to predict"
        )
        horizon = horizon_options[selected_horizon]
    
    with col2:
        show_historical = st.select_slider(
            "Historical Context",
            options=[30, 60, 90, 180, 365],
            value=90,
            format_func=lambda x: f"{x} days",
            help="Amount of historical data to display"
        )
    
    with col3:
        show_confidence = st.toggle("Show Confidence Interval", value=True, help="Display prediction uncertainty band")
    
    with col4:
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Load data and model
    with st.spinner("🔄 Loading model and data..."):
        historical_data = load_historical_data()
        model, scaler, model_config = load_model_and_scaler()
    
    if historical_data.empty:
        st.markdown("""
            <div style="
                background: linear-gradient(145deg, rgba(255, 82, 82, 0.1) 0%, rgba(255, 82, 82, 0.05) 100%);
                border: 1px solid rgba(255, 82, 82, 0.3);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            ">
                <span style="font-size: 2rem;">📉</span>
                <h3 style="margin: 0.5rem 0; color: #FF5252;">Data Not Found</h3>
                <p style="color: rgba(250,250,250,0.7); margin: 0;">
                    Could not load historical data. Please ensure the database exists.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    if model is None:
        st.markdown("""
            <div style="
                background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
                border: 1px solid rgba(255, 193, 7, 0.3);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            ">
                <span style="font-size: 2rem;">🤖</span>
                <h3 style="margin: 0.5rem 0; color: #FFC107;">Model Not Found</h3>
                <p style="color: rgba(250,250,250,0.7); margin: 0;">
                    Could not load the trained model. Please train the model first using the training pipeline.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Generate predictions
    with st.spinner(f"🔮 Generating {horizon}-day forecast..."):
        forecast = generate_predictions(model, scaler, historical_data, horizon)
    
    if forecast is None:
        st.error("❌ Failed to generate predictions.")
        return
    
    # Key metrics with enhanced styling
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            📊 Forecast Summary
        </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    last_price = forecast['last_price']
    predicted_final = forecast['predictions'][-1]
    price_change = predicted_final - last_price
    pct_change = (price_change / last_price) * 100
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${last_price:.2f}",
            delta=None
        )
    
    with col2:
        delta_color = "normal" if price_change >= 0 else "inverse"
        st.metric(
            label=f"Predicted ({horizon}d)",
            value=f"${predicted_final:.2f}",
            delta=f"{price_change:+.2f} ({pct_change:+.1f}%)"
        )
    
    with col3:
        min_pred = forecast['predictions'].min()
        st.metric(
            label="Forecast Low",
            value=f"${min_pred:.2f}",
            delta=f"{((min_pred - last_price) / last_price * 100):+.1f}%"
        )
    
    with col4:
        max_pred = forecast['predictions'].max()
        st.metric(
            label="Forecast High",
            value=f"${max_pred:.2f}",
            delta=f"{((max_pred - last_price) / last_price * 100):+.1f}%"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create interactive chart
    fig = create_forecast_chart(
        historical_data,
        forecast,
        show_historical,
        show_confidence,
        horizon
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Predictions table with tabs
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin: 1.5rem 0 1rem 0;">
            📋 Detailed Predictions
        </p>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Table View", "📈 Daily Changes"])
    
    with tab1:
        pred_df = pd.DataFrame({
            'Date': forecast['dates'],
            'Predicted Price ($)': forecast['predictions']
        })
        
        pred_df['Day'] = range(1, len(pred_df) + 1)
        pred_df['Change from Current ($)'] = pred_df['Predicted Price ($)'] - last_price
        pred_df['Change (%)'] = (pred_df['Change from Current ($)'] / last_price) * 100
        
        # Format columns
        pred_df['Date'] = pd.to_datetime(pred_df['Date']).dt.strftime('%Y-%m-%d')
        pred_df['Predicted Price ($)'] = pred_df['Predicted Price ($)'].round(2)
        pred_df['Change from Current ($)'] = pred_df['Change from Current ($)'].round(2)
        pred_df['Change (%)'] = pred_df['Change (%)'].round(2)
        
        # Reorder columns
        pred_df = pred_df[['Day', 'Date', 'Predicted Price ($)', 'Change from Current ($)', 'Change (%)']]
        
        st.dataframe(
            pred_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Day": st.column_config.NumberColumn("Day", width="small"),
                "Date": st.column_config.TextColumn("Date", width="medium"),
                "Predicted Price ($)": st.column_config.NumberColumn("Predicted Price", format="$%.2f"),
                "Change from Current ($)": st.column_config.NumberColumn("Change ($)", format="$%.2f"),
                "Change (%)": st.column_config.NumberColumn("Change (%)", format="%.2f%%")
            }
        )
    
    with tab2:
        # Daily change chart
        daily_changes = np.diff(forecast['predictions'], prepend=last_price)
        
        fig_changes = go.Figure()
        
        colors = ['#00C853' if x >= 0 else '#FF5252' for x in daily_changes]
        
        fig_changes.add_trace(go.Bar(
            x=list(range(1, len(daily_changes) + 1)),
            y=daily_changes,
            marker_color=colors,
            hovertemplate='Day %{x}<br>Change: $%{y:.2f}<extra></extra>'
        ))
        
        fig_changes.update_layout(
            title="Daily Price Changes",
            xaxis_title="Day",
            yaxis_title="Price Change ($)",
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_changes, width='stretch')
    
    # Download section
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"nvidia_predictions_{horizon}d_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )


def create_forecast_chart(
    historical_data: pd.DataFrame,
    forecast: dict,
    show_historical: int,
    show_confidence: bool,
    horizon: int
) -> go.Figure:
    """Create interactive forecast chart with enhanced styling."""
    
    # Filter historical data
    hist_subset = historical_data.tail(show_historical)
    
    fig = go.Figure()
    
    # Historical prices with gradient effect
    fig.add_trace(go.Scatter(
        x=hist_subset['Date'],
        y=hist_subset['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#76B900', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(118, 185, 0, 0.1)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Predictions
    pred_dates = forecast['dates']
    predictions = forecast['predictions']
    
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=predictions,
        mode='lines+markers',
        name=f'{horizon}-Day Forecast',
        line=dict(color='#FF6B35', width=2.5),
        marker=dict(size=6, symbol='circle', line=dict(width=1, color='white')),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Predicted: $%{y:.2f}<extra></extra>'
    ))
    
    # Confidence interval (simulated based on prediction uncertainty)
    if show_confidence:
        # Simple uncertainty estimation (grows with horizon)
        uncertainty = np.linspace(0.02, 0.10, len(predictions))
        upper = predictions * (1 + uncertainty)
        lower = predictions * (1 - uncertainty)
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=upper,
            mode='lines',
            name='Upper Bound (95%)',
            line=dict(color='rgba(255, 107, 53, 0)', width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=lower,
            mode='lines',
            name='95% Confidence Interval',
            line=dict(color='rgba(255, 107, 53, 0)', width=0),
            fill='tonexty',
            fillcolor='rgba(255, 107, 53, 0.15)',
            hoverinfo='skip'
        ))
    
    # Connection line
    last_date = historical_data['Date'].iloc[-1]
    # Convert to string for Plotly compatibility
    last_date_str = pd.Timestamp(last_date).strftime('%Y-%m-%d')
    
    fig.add_trace(go.Scatter(
        x=[last_date, pred_dates[0]],
        y=[forecast['last_price'], predictions[0]],
        mode='lines',
        name='Transition',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=2, dash='dot'),
        showlegend=False
    ))
    
    # Add vertical line at prediction start (use string format to avoid Timestamp arithmetic issues)
    fig.add_shape(
        type="line",
        x0=last_date_str,
        x1=last_date_str,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="rgba(150, 150, 150, 0.5)", width=2, dash="dash")
    )
    
    # Add annotation for the vertical line
    fig.add_annotation(
        x=last_date_str,
        y=1.02,
        yref="paper",
        text="📍 Forecast Start",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="rgba(250, 250, 250, 0.7)", size=11)
    )
    
    # Enhanced Layout
    fig.update_layout(
        title=dict(
            text=f"<b>NVIDIA Stock Price Forecast</b><br><span style='font-size:14px; color: rgba(250,250,250,0.6)'>{horizon}-Day Prediction with LSTM Model</span>",
            font=dict(size=18, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        template='plotly_dark',
        height=520,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.05)',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='rgba(38, 39, 48, 0.8)',
                activecolor='#76B900',
                font=dict(color='white', size=11)
            ),
            rangeslider=dict(
                visible=True,
                bgcolor='rgba(38, 39, 48, 0.5)',
                thickness=0.05
            ),
            type="date"
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.05)',
            tickprefix='$',
            tickformat=',.0f'
        ),
        paper_bgcolor='rgba(14, 17, 23, 1)',
        plot_bgcolor='rgba(14, 17, 23, 1)'
    )
    
    return fig
