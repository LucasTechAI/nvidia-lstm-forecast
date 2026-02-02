"""
Main Streamlit Dashboard Application.

NVIDIA Stock Price Prediction Dashboard.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.components.predictions import render_predictions_page
from src.dashboard.components.metrics import render_metrics_page
from src.dashboard.components.model_schema import render_model_schema_page
from src.dashboard.components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="NVIDIA Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced UI/UX
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #76B900 0%, #9ED700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        text-align: center;
        color: rgba(250, 250, 250, 0.6);
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(145deg, #1a1c24 0%, #262730 100%);
        border: 1px solid rgba(118, 185, 0, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        border-color: rgba(118, 185, 0, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(118, 185, 0, 0.15);
    }
    
    /* Enhanced Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1c24 0%, #262730 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: rgba(118, 185, 0, 0.4);
        box-shadow: 0 4px 20px rgba(118, 185, 0, 0.1);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: rgba(250, 250, 250, 0.7) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #76B900 !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(118, 185, 0, 0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(145deg, rgba(118, 185, 0, 0.1) 0%, rgba(118, 185, 0, 0.05) 100%);
        border: 1px solid rgba(118, 185, 0, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 1rem 0;
    }
    
    .info-box-title {
        font-weight: 600;
        color: #76B900;
        margin-bottom: 0.5rem;
    }
    
    /* Success/Warning/Error boxes */
    .success-box {
        background: linear-gradient(145deg, rgba(0, 200, 83, 0.1) 0%, rgba(0, 200, 83, 0.05) 100%);
        border: 1px solid rgba(0, 200, 83, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
    }
    
    .warning-box {
        background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #76B900 0%, #5a8f00 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(118, 185, 0, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #8ed100 0%, #76B900 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(118, 185, 0, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #42A5F5 0%, #2196F3 100%);
        transform: translateY(-1px);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background-color: #1a1c24;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stSlider > div > div {
        background-color: rgba(118, 185, 0, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1a1c24 0%, #262730 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: rgba(118, 185, 0, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1c24;
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom: none;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #262730 0%, #1a1c24 100%);
        border-color: rgba(118, 185, 0, 0.5);
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(118, 185, 0, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #76B900 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Tooltip styles */
    .tooltip {
        background: #1a1c24 !important;
        border: 1px solid rgba(118, 185, 0, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1c24;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(118, 185, 0, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(118, 185, 0, 0.7);
    }
    
    /* Prediction cards */
    .prediction-up {
        color: #00C853;
        font-weight: 600;
    }
    
    .prediction-down {
        color: #FF5252;
        font-weight: 600;
    }
    
    /* Status indicator */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #00C853;
        box-shadow: 0 0 8px rgba(0, 200, 83, 0.5);
    }
    
    .status-inactive {
        background-color: #FF5252;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        
        [data-testid="stMetric"] {
            padding: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Main header with subtitle
    st.markdown('<p class="main-header">📈 NVIDIA Stock Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Stock Price Forecasting with LSTM Neural Networks</p>', unsafe_allow_html=True)
    
    # Render selected page
    if selected_page == "Predictions":
        render_predictions_page()
    elif selected_page == "Model Metrics":
        render_metrics_page()
    elif selected_page == "Model Schema":
        render_model_schema_page()
    else:
        render_predictions_page()  # Default page


if __name__ == "__main__":
    main()
