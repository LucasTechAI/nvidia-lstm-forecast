"""
Model Schema Component for Dashboard.

Displays model architecture and configuration details.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent.parent.parent


def load_model_info() -> dict:
    """Load model information from checkpoint."""
    checkpoint_path = project_root / "data" / "models" / "checkpoints" / "best_model.pt"
    
    if not checkpoint_path.exists():
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def count_parameters(model_state_dict: dict) -> dict:
    """Count model parameters by layer."""
    param_counts = {}
    total_params = 0
    trainable_params = 0
    
    for name, param in model_state_dict.items():
        count = param.numel()
        param_counts[name] = {
            'shape': list(param.shape),
            'count': count,
            'dtype': str(param.dtype)
        }
        total_params += count
        trainable_params += count
    
    return {
        'layers': param_counts,
        'total': total_params,
        'trainable': trainable_params
    }


def render_model_tree(model_config: dict, model_state: dict):
    """Render an interactive tree view of the model architecture."""
    
    hidden_size = model_config.get('hidden_size', 128)
    num_layers = model_config.get('num_layers', 2)
    input_size = model_config.get('input_size', 1)
    output_size = model_config.get('output_size', 1)
    dropout = model_config.get('dropout', 0.2)
    bidirectional = model_config.get('bidirectional', False)
    seq_length = model_config.get('sequence_length', 60)
    
    # Count parameters per component
    lstm_params = 0
    fc_params = 0
    if model_state:
        for name, info in model_state.items():
            param_count = info.numel() if hasattr(info, 'numel') else 0
            if 'lstm' in name.lower():
                lstm_params += param_count
            elif 'fc' in name.lower() or 'linear' in name.lower():
                fc_params += param_count
    
    total_params = lstm_params + fc_params
    
    # Build tree structure as HTML
    tree_html = f"""
    <div style="
        background: linear-gradient(145deg, #1a1c24, #262730);
        border: 1px solid rgba(118, 185, 0, 0.3);
        border-radius: 12px;
        padding: 24px;
        font-family: 'Consolas', 'Monaco', monospace;
    ">
        <div style="color: #76B900; font-weight: bold; font-size: 1.1rem; margin-bottom: 1rem;">
            📦 NvidiaLSTM
        </div>
        <div style="margin-left: 20px; border-left: 2px solid rgba(118, 185, 0, 0.3); padding-left: 20px;">
            <!-- Input Layer -->
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #4ECDC4;">📥</span>
                    <span style="color: #4ECDC4; font-weight: 600;">Input Layer</span>
                    <span style="
                        background: rgba(78, 205, 196, 0.2);
                        color: #4ECDC4;
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 0.75rem;
                    ">shape: (batch, {seq_length}, {input_size})</span>
                </div>
            </div>
            <!-- LSTM Block -->
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
                    <span style="color: #76B900;">🧠</span>
                    <span style="color: #76B900; font-weight: 600;">LSTM</span>
                    <span style="
                        background: rgba(118, 185, 0, 0.2);
                        color: #76B900;
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 0.75rem;
                    ">{lstm_params:,} params</span>
                </div>
                <div style="margin-left: 28px; border-left: 2px solid rgba(118, 185, 0, 0.2); padding-left: 16px;">
                    <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem; margin-bottom: 4px;">
                        ├─ <span style="color: #9ED700;">input_size:</span> {input_size}
                    </div>
                    <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem; margin-bottom: 4px;">
                        ├─ <span style="color: #9ED700;">hidden_size:</span> {hidden_size}
                    </div>
                    <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem; margin-bottom: 4px;">
                        ├─ <span style="color: #9ED700;">num_layers:</span> {num_layers}
                    </div>
                    <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem; margin-bottom: 4px;">
                        ├─ <span style="color: #9ED700;">bidirectional:</span> {str(bidirectional)}
                    </div>
                    <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem; margin-bottom: 4px;">
                        ├─ <span style="color: #9ED700;">batch_first:</span> True
                    </div>
                    <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem;">
                        └─ <span style="color: #9ED700;">dropout:</span> {dropout}
                    </div>
                </div>
                <!-- LSTM Layers Detail -->
                <div style="margin-left: 28px; margin-top: 0.75rem;">
    """
    
    # Add individual LSTM layer details
    for i in range(num_layers):
        layer_color = "#76B900" if i % 2 == 0 else "#9ED700"
        is_last = i == num_layers - 1
        connector = "└─" if is_last else "├─"
        tree_html += f"""
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                        <span style="color: rgba(250,250,250,0.5);">{connector}</span>
                        <span style="color: {layer_color};">Layer {i}</span>
                        <span style="color: rgba(250,250,250,0.5); font-size: 0.75rem;">
                            ({input_size if i == 0 else hidden_size} → {hidden_size})
                        </span>
                    </div>
        """
    
    tree_html += f"""
                </div>
            </div>
            <!-- Dropout Layer -->
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #45B7D1;">💧</span>
                    <span style="color: #45B7D1; font-weight: 600;">Dropout</span>
                    <span style="
                        background: rgba(69, 183, 209, 0.2);
                        color: #45B7D1;
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 0.75rem;
                    ">p={dropout}</span>
                </div>
            </div>
            <!-- Fully Connected Layer -->
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
                    <span style="color: #FF6B35;">🔗</span>
                    <span style="color: #FF6B35; font-weight: 600;">Linear (FC)</span>
                    <span style="
                        background: rgba(255, 107, 53, 0.2);
                        color: #FF6B35;
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 0.75rem;
                    ">{fc_params:,} params</span>
                </div>
                <div style="margin-left: 28px; border-left: 2px solid rgba(255, 107, 53, 0.2); padding-left: 16px;">
                    <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem; margin-bottom: 4px;">
                        ├─ <span style="color: #FF8C5A;">in_features:</span> {hidden_size}
                    </div>
                    <div style="color: rgba(250,250,250,0.7); font-size: 0.85rem;">
                        └─ <span style="color: #FF8C5A;">out_features:</span> {output_size}
                    </div>
                </div>
            </div>
            <!-- Output Layer -->
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #96CEB4;">📤</span>
                    <span style="color: #96CEB4; font-weight: 600;">Output</span>
                    <span style="
                        background: rgba(150, 206, 180, 0.2);
                        color: #96CEB4;
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 0.75rem;
                    ">shape: (batch, {output_size})</span>
                </div>
            </div>
        </div>
        <!-- Summary -->
        <div style="
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div style="color: rgba(250,250,250,0.5); font-size: 0.8rem;">
                Total Parameters: <span style="color: #76B900; font-weight: 600;">{total_params:,}</span>
            </div>
            <div style="
                background: rgba(118, 185, 0, 0.1);
                border: 1px solid rgba(118, 185, 0, 0.3);
                border-radius: 6px;
                padding: 4px 12px;
                font-size: 0.75rem;
                color: #76B900;
            ">
                PyTorch LSTM
            </div>
        </div>
    </div>
    """
    tree_html = tree_html.replace('\n', '').replace('    ', '')
    st.markdown(tree_html, unsafe_allow_html=True)


def render_model_schema_page():
    """Render the model schema page."""
    
    # Enhanced page header
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-weight: 600;">🧠 Model Architecture</h2>
            <p style="color: rgba(250,250,250,0.6); margin-top: 0.5rem;">
                Explore the LSTM neural network architecture, layer configuration, and parameter distribution.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model info
    with st.spinner("🔄 Loading model information..."):
        checkpoint = load_model_info()
    
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
                    Please train the model first to view architecture details.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    model_config = checkpoint.get('model_config', {})
    model_state = checkpoint.get('model_state_dict', {})
    
    # Model Overview with styled cards
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            📋 Model Overview
        </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="
                background: linear-gradient(145deg, #1a1c24, #262730);
                border: 1px solid rgba(118, 185, 0, 0.2);
                border-radius: 12px;
                padding: 20px;
                height: 100%;
            ">
                <h4 style="color: #76B900; margin: 0 0 0.5rem 0; font-size: 1rem;">🔷 Architecture Type</h4>
                <p style="font-weight: 600; color: white; margin: 0 0 0.5rem 0;">LSTM (Long Short-Term Memory)</p>
                <p style="color: rgba(250,250,250,0.6); font-size: 0.9rem; margin: 0;">
                    A recurrent neural network capable of learning long-term dependencies in sequential data.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style="
                background: linear-gradient(145deg, #1a1c24, #262730);
                border: 1px solid rgba(0, 200, 83, 0.2);
                border-radius: 12px;
                padding: 20px;
            ">
                <h4 style="color: #00C853; margin: 0 0 0.5rem 0; font-size: 1rem;">🎯 Use Case</h4>
                <p style="font-weight: 600; color: white; margin: 0 0 0.5rem 0;">Time Series Forecasting</p>
                <p style="color: rgba(250,250,250,0.6); font-size: 0.9rem; margin: 0;">
                    Predicting future NVIDIA stock prices based on historical patterns.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="
                background: linear-gradient(145deg, #1a1c24, #262730);
                border: 1px solid rgba(33, 150, 243, 0.2);
                border-radius: 12px;
                padding: 20px;
                height: 100%;
            ">
                <h4 style="color: #2196F3; margin: 0 0 1rem 0; font-size: 1rem;">⚡ Key Characteristics</h4>
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #76B900;">▸</span>
                        <span style="color: rgba(250,250,250,0.8);"><b>Sequential Processing</b> — Processes data one step at a time</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #76B900;">▸</span>
                        <span style="color: rgba(250,250,250,0.8);"><b>Memory Cells</b> — Maintains information over long sequences</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #76B900;">▸</span>
                        <span style="color: rgba(250,250,250,0.8);"><b>Gating Mechanism</b> — Input, Forget, Output gates</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #76B900;">▸</span>
                        <span style="color: rgba(250,250,250,0.8);"><b>Bidirectional</b> — Optional dual-direction processing</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Architecture Configuration
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            ⚙️ Architecture Configuration
        </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Input Size", model_config.get('input_size', 1))
        st.caption("Number of input features")
    
    with col2:
        st.metric("Hidden Size", model_config.get('hidden_size', 128))
        st.caption("LSTM hidden state dimension")
    
    with col3:
        st.metric("Num Layers", model_config.get('num_layers', 2))
        st.caption("Stacked LSTM layers")
    
    with col4:
        st.metric("Output Size", model_config.get('output_size', 1))
        st.caption("Prediction dimension")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dropout", f"{model_config.get('dropout', 0.2):.0%}")
        st.caption("Regularization rate")
    
    with col2:
        bidirectional = model_config.get('bidirectional', False)
        st.metric("Bidirectional", "Yes ✅" if bidirectional else "No ❌")
        st.caption("Process both directions")
    
    with col3:
        seq_length = model_config.get('sequence_length', 60)
        st.metric("Sequence Length", seq_length)
        st.caption("Input window size (days)")
    
    with col4:
        num_directions = 2 if bidirectional else 1
        st.metric("Directions", num_directions)
        st.caption("Processing directions")
    
    st.markdown("---")
    
    # Model Tree View
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            🌳 Model Tree Structure
        </p>
    """, unsafe_allow_html=True)
    
    render_model_tree(model_config, model_state)
    
    st.markdown("---")
    
    # Architecture Visualization
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            🎨 Architecture Visualization
        </p>
    """, unsafe_allow_html=True)
    
    render_architecture_diagram(model_config)
    
    st.markdown("---")
    
    # Parameter Analysis
    st.markdown("""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            📊 Parameter Analysis
        </p>
    """, unsafe_allow_html=True)
    
    if model_state:
        param_info = count_parameters(model_state)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Parameters",
                f"{param_info['total']:,}",
                help="Total number of learnable parameters"
            )
        
        with col2:
            st.metric(
                "Trainable Parameters",
                f"{param_info['trainable']:,}",
                help="Parameters updated during training"
            )
        
        with col3:
            size_mb = param_info['total'] * 4 / (1024 * 1024)  # float32 = 4 bytes
            st.metric(
                "Model Size",
                f"{size_mb:.2f} MB",
                help="Approximate model size in memory"
            )
        
        # Layer breakdown
        st.markdown("#### Layer-wise Parameter Distribution")
        
        render_parameter_distribution(param_info)
        
        # Detailed layer table
        with st.expander("📋 Detailed Layer Information"):
            layer_data = []
            for name, info in param_info['layers'].items():
                layer_data.append({
                    'Layer': name,
                    'Shape': str(info['shape']),
                    'Parameters': f"{info['count']:,}",
                    'Type': info['dtype']
                })
            
            layer_df = pd.DataFrame(layer_data)
            st.dataframe(layer_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Data Flow
    st.markdown("### 🔄 Data Flow")
    
    render_data_flow(model_config)
    
    st.markdown("---")
    
    # Training Configuration
    st.markdown("### 🎯 Training Configuration")
    
    training_info = checkpoint.get('training_info', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Optimizer & Learning")
        
        optimizer_config = {
            'Optimizer': training_info.get('optimizer', 'Adam'),
            'Learning Rate': training_info.get('learning_rate', 0.001),
            'Weight Decay': training_info.get('weight_decay', 1e-5),
            'Batch Size': training_info.get('batch_size', 32)
        }
        
        for key, value in optimizer_config.items():
            if isinstance(value, float) and value < 0.01:
                st.markdown(f"**{key}:** `{value:.2e}`")
            else:
                st.markdown(f"**{key}:** `{value}`")
    
    with col2:
        st.markdown("#### Regularization & Stopping")
        
        reg_config = {
            'Dropout Rate': model_config.get('dropout', 0.2),
            'Early Stopping Patience': training_info.get('early_stopping_patience', 10),
            'Gradient Clipping': training_info.get('gradient_clip_value', 1.0),
            'LR Scheduler': training_info.get('use_scheduler', True)
        }
        
        for key, value in reg_config.items():
            if isinstance(value, bool):
                st.markdown(f"**{key}:** `{'Yes' if value else 'No'}`")
            elif isinstance(value, float):
                st.markdown(f"**{key}:** `{value:.2f}`")
            else:
                st.markdown(f"**{key}:** `{value}`")
    
    st.markdown("---")
    
    # Export Model Info
    st.markdown("### 💾 Export Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_summary = {
            'architecture': 'LSTM',
            'config': model_config,
            'training_info': training_info,
            'total_parameters': param_info['total'] if model_state else 0
        }
        
        import json
        json_str = json.dumps(model_summary, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Model Config (JSON)",
            data=json_str,
            file_name="model_config.json",
            mime="application/json"
        )
    
    with col2:
        # Generate ONNX-like description
        onnx_desc = f"""
Model: NvidiaLSTM
Input: (batch_size, {model_config.get('sequence_length', 60)}, {model_config.get('input_size', 1)})
Output: (batch_size, {model_config.get('output_size', 1)})

Layers:
  - LSTM: input={model_config.get('input_size', 1)}, hidden={model_config.get('hidden_size', 128)}, layers={model_config.get('num_layers', 2)}
  - Dropout: p={model_config.get('dropout', 0.2)}
  - Linear: in={model_config.get('hidden_size', 128)}, out={model_config.get('output_size', 1)}

Total Parameters: {param_info['total'] if model_state else 'N/A':,}
"""
        
        st.download_button(
            label="📥 Download Model Summary (TXT)",
            data=onnx_desc,
            file_name="model_summary.txt",
            mime="text/plain"
        )


def render_architecture_diagram(config: dict):
    """Render architecture diagram using Plotly."""
    
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 2)
    seq_length = config.get('sequence_length', 60)
    bidirectional = config.get('bidirectional', False)
    
    fig = go.Figure()
    
    # Layer positions
    layers = ['Input', 'LSTM\nLayers', 'Dropout', 'Dense', 'Output']
    x_positions = [0, 1, 2, 3, 4]
    
    # Layer sizes for visualization
    sizes = [40, 80, 60, 50, 30]
    colors = ['#4ECDC4', '#76B900', '#45B7D1', '#FF6B35', '#96CEB4']
    
    # Add nodes
    for i, (layer, x, size, color) in enumerate(zip(layers, x_positions, sizes, colors)):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[0],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                line=dict(width=2, color='white')
            ),
            text=[layer],
            textposition='bottom center',
            textfont=dict(size=12, color='white'),
            hoverinfo='text',
            hovertext=get_layer_description(layer, config),
            showlegend=False
        ))
    
    # Add connections
    for i in range(len(x_positions) - 1):
        fig.add_trace(go.Scatter(
            x=[x_positions[i] + 0.15, x_positions[i+1] - 0.15],
            y=[0, 0],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add layer details as annotations
    annotations = [
        dict(x=0, y=0.5, text=f"({seq_length}, 1)", showarrow=False, font=dict(size=10, color='gray')),
        dict(x=1, y=0.5, text=f"{num_layers}x LSTM\n({hidden_size} units)", showarrow=False, font=dict(size=10, color='gray')),
        dict(x=2, y=0.5, text=f"p={config.get('dropout', 0.2)}", showarrow=False, font=dict(size=10, color='gray')),
        dict(x=3, y=0.5, text=f"{hidden_size}→1", showarrow=False, font=dict(size=10, color='gray')),
        dict(x=4, y=0.5, text="(1,)", showarrow=False, font=dict(size=10, color='gray'))
    ]
    
    if bidirectional:
        annotations.append(
            dict(x=1, y=-0.5, text="Bidirectional ↔", showarrow=False, font=dict(size=10, color='#76B900'))
        )
    
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 4.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1, 1]
        ),
        template='plotly_dark',
        height=250,
        margin=dict(l=20, r=20, t=20, b=60),
        title=dict(
            text='Neural Network Architecture',
            font=dict(size=16),
            x=0.5
        )
    )
    
    st.plotly_chart(fig, width='stretch')


def get_layer_description(layer: str, config: dict) -> str:
    """Get description for each layer."""
    descriptions = {
        'Input': f"Input Layer\nShape: (batch, {config.get('sequence_length', 60)}, {config.get('input_size', 1)})\nReceives normalized stock prices",
        'LSTM\nLayers': f"LSTM Layers\n{config.get('num_layers', 2)} stacked layers\n{config.get('hidden_size', 128)} hidden units\nLearns temporal patterns",
        'Dropout': f"Dropout Layer\nRate: {config.get('dropout', 0.2)}\nPrevents overfitting",
        'Dense': f"Dense Layer\nFully connected\nMaps to output",
        'Output': "Output Layer\nSingle value\nPredicted price (normalized)"
    }
    return descriptions.get(layer, layer)


def render_parameter_distribution(param_info: dict):
    """Render parameter distribution chart."""
    
    # Group by layer type
    layer_groups = {
        'LSTM Weights (ih)': 0,
        'LSTM Weights (hh)': 0,
        'LSTM Biases': 0,
        'Dense Layer': 0
    }
    
    for name, info in param_info['layers'].items():
        count = info['count']
        if 'weight_ih' in name:
            layer_groups['LSTM Weights (ih)'] += count
        elif 'weight_hh' in name:
            layer_groups['LSTM Weights (hh)'] += count
        elif 'bias' in name and 'lstm' in name.lower():
            layer_groups['LSTM Biases'] += count
        else:
            layer_groups['Dense Layer'] += count
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(layer_groups.keys()),
        values=list(layer_groups.values()),
        hole=0.4,
        marker=dict(colors=['#76B900', '#4ECDC4', '#45B7D1', '#FF6B35']),
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        template='plotly_dark',
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=[dict(
            text=f'{param_info["total"]:,}',
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )],
        title=dict(
            text='Parameter Distribution by Layer Type',
            font=dict(size=14)
        )
    )
    
    st.plotly_chart(fig, width='stretch')


def render_data_flow(config: dict):
    """Render data flow description."""
    
    seq_length = config.get('sequence_length', 60)
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 2)
    
    flow_steps = [
        {
            'step': '1. Input Preparation',
            'description': f'Historical prices are normalized and shaped into sequences of {seq_length} time steps.',
            'shape': f'(batch_size, {seq_length}, 1)'
        },
        {
            'step': '2. LSTM Processing',
            'description': f'{num_layers} stacked LSTM layers process the sequence, learning temporal patterns.',
            'shape': f'(batch_size, {seq_length}, {hidden_size})'
        },
        {
            'step': '3. Final Hidden State',
            'description': 'The last hidden state from the final LSTM layer is extracted.',
            'shape': f'(batch_size, {hidden_size})'
        },
        {
            'step': '4. Dropout',
            'description': f'Dropout with rate {config.get("dropout", 0.2)} is applied for regularization.',
            'shape': f'(batch_size, {hidden_size})'
        },
        {
            'step': '5. Dense Layer',
            'description': 'Fully connected layer maps hidden state to prediction.',
            'shape': '(batch_size, 1)'
        },
        {
            'step': '6. Inverse Transform',
            'description': 'Prediction is converted back to original price scale.',
            'shape': 'Predicted Price ($)'
        }
    ]
    
    for step in flow_steps:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{step['step']}**")
            st.markdown(f"_{step['description']}_")
        with col2:
            st.code(step['shape'])
        st.markdown("")
