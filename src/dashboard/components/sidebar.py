"""
Sidebar Component for Dashboard Navigation.
"""

import streamlit as st
from datetime import datetime


def render_sidebar() -> str:
    """
    Render the sidebar with navigation and settings.
    
    Returns:
        Selected page name.
    """
    with st.sidebar:
        # Logo with custom styling
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <img src="https://upload.wikimedia.org/wikipedia/sco/thumb/2/21/Nvidia_logo.svg/1280px-Nvidia_logo.svg.png" 
                     width="180" style="filter: brightness(1.1);">
            </div>
        """, unsafe_allow_html=True)
        
        # Status indicator
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <span style="
                    display: inline-flex;
                    align-items: center;
                    background: rgba(0, 200, 83, 0.1);
                    border: 1px solid rgba(0, 200, 83, 0.3);
                    border-radius: 20px;
                    padding: 4px 12px;
                    font-size: 0.75rem;
                    color: #00C853;
                ">
                    <span style="
                        width: 6px;
                        height: 6px;
                        background: #00C853;
                        border-radius: 50%;
                        margin-right: 6px;
                        box-shadow: 0 0 6px #00C853;
                    "></span>
                    Model Active
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation section with enhanced styling
        st.markdown("""
            <p style="
                color: rgba(250, 250, 250, 0.5);
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                margin-bottom: 0.5rem;
            ">Navigation</p>
        """, unsafe_allow_html=True)
        
        pages = {
            "📊 Stock Predictions": "Predictions",
            "📈 Model Metrics": "Model Metrics", 
            "🧠 Model Architecture": "Model Schema"
        }
        
        selected = st.radio(
            "Select a page:",
            list(pages.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("""
            <p style="
                color: rgba(250, 250, 250, 0.5);
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                margin-bottom: 0.5rem;
            ">Quick Stats</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div style="
                    background: linear-gradient(145deg, #1a1c24, #262730);
                    border: 1px solid rgba(118, 185, 0, 0.2);
                    border-radius: 8px;
                    padding: 12px;
                    text-align: center;
                ">
                    <div style="font-size: 0.7rem; color: rgba(250,250,250,0.5); text-transform: uppercase;">Accuracy</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #76B900;">95.1%</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div style="
                    background: linear-gradient(145deg, #1a1c24, #262730);
                    border: 1px solid rgba(118, 185, 0, 0.2);
                    border-radius: 8px;
                    padding: 12px;
                    text-align: center;
                ">
                    <div style="font-size: 0.7rem; color: rgba(250,250,250,0.5); text-transform: uppercase;">R² Score</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #76B900;">0.91</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # About section with expander
        with st.expander("ℹ️ About this Dashboard", expanded=False):
            st.markdown("""
            **NVIDIA Stock Predictor** uses advanced LSTM neural networks to forecast stock prices.
            
            **Features:**
            - 🔮 Multi-horizon forecasting
            - 📊 Real-time performance metrics
            - 🏗️ Model architecture visualization
            - 📥 Data export capabilities
            
            **Tech Stack:**
            - PyTorch & LSTM
            - MLflow tracking
            - Streamlit dashboard
            """)
        
        st.markdown("---")
        
        # Last updated timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div style="font-size: 0.65rem; color: rgba(250,250,250,0.4); text-transform: uppercase; letter-spacing: 1px;">
                    Last Updated
                </div>
                <div style="font-size: 0.8rem; color: rgba(250,250,250,0.7);">
                    {current_time}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
            <div style="
                text-align: center;
                padding: 1rem 0;
                margin-top: 1rem;
                border-top: 1px solid rgba(255,255,255,0.1);
            ">
                <div style="font-size: 0.7rem; color: rgba(250,250,250,0.4);">
                    Built with ❤️ using
                </div>
                <div style="font-size: 0.8rem; color: rgba(250,250,250,0.6); margin-top: 4px;">
                    Streamlit • PyTorch • MLflow
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    return pages[selected]
