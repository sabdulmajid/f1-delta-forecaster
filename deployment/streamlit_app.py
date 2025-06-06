"""
Streamlit app for F1 Tyre-Degradation Forecaster.
Interactive widget to explore model predictions.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

@st.cache_data
def load_data():
    """Load processed F1 data."""
    data_path = "data/processed/f1_data_2023.pkl"
    if not Path(data_path).exists():
        st.error(f"Data file not found: {data_path}")
        st.info("Please run data processing first: `python data/data_loader.py --year 2023`")
        return None
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

@st.cache_resource
def load_model():
    """Load trained transformer model."""
    model_paths = [
        "models/checkpoints/lightning_logs/version_0/checkpoints/last.ckpt",
        "models/checkpoints/f1-forecaster-best.ckpt",
        "cluster_results/checkpoints/last.ckpt"
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            try:
                from training.lightning_module import F1LightningModule
                model = F1LightningModule.load_from_checkpoint(model_path)
                model.eval()
                return model, model_path
            except Exception as e:
                st.warning(f"Failed to load model from {model_path}: {e}")
                continue
    
    return None, None

def create_sample_sequence(
    speed_profile: str,
    tyre_age: int,
    compound: str,
    weather_temp: float,
    sequence_length: int = 5
) -> np.ndarray:
    """Create a sample sequence based on user inputs."""
    
    # Speed profiles
    speed_profiles = {
        "Conservative": {"base": 250, "variation": 15},
        "Aggressive": {"base": 280, "variation": 25},
        "Consistent": {"base": 265, "variation": 8}
    }
    
    # Compound mappings
    compound_mapping = {
        "SOFT": [1, 0, 0],
        "MEDIUM": [0, 1, 0],
        "HARD": [0, 0, 1]
    }
    
    profile = speed_profiles[speed_profile]
    base_speed = profile["base"]
    variation = profile["variation"]
    
    # Generate sequence
    sequence = []
    for i in range(sequence_length):
        # Speed features with some variation
        speed_mean = base_speed + np.random.normal(0, variation)
        speed_max = speed_mean + np.random.uniform(10, 30)
        speed_min = speed_mean - np.random.uniform(10, 20)
        speed_std = np.random.uniform(5, 15)
        
        # Throttle features
        throttle_mean = np.random.uniform(60, 85)
        throttle_max = 100
        throttle_time_full = np.random.uniform(0.2, 0.6)
        
        # Other features
        brake_mean = np.random.uniform(0, 20)
        brake_time_active = np.random.uniform(0.1, 0.3)
        drs_active = np.random.uniform(0, 0.8)
        gear_changes = np.random.randint(0, 5)
        turns_since_start = i * 5  # Micro-sector progression
        
        # Weather
        air_temp = weather_temp
        track_temp = weather_temp + 10
        humidity = np.random.uniform(40, 70)
        
        # Create feature vector
        features = [
            speed_mean, speed_max, speed_min, speed_std,
            throttle_mean, throttle_max, throttle_time_full,
            brake_mean, brake_time_active, drs_active,
            gear_changes, turns_since_start, tyre_age,
            air_temp, track_temp, humidity
        ]
        
        # Add compound features
        features.extend(compound_mapping[compound])
        
        sequence.append(features)
    
    return np.array(sequence)

def main():
    st.set_page_config(
        page_title="F1 Tyre-Degradation Forecaster",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title("🏎️ F1 Tyre-Degradation Forecaster")
    st.markdown("Interactive prediction of Formula 1 lap pace deltas based on tyre degradation and micro-sector data.")
    
    # Load data and model
    data = load_data()
    model, model_path = load_model()
    
    if data is None:
        return
    
    # Sidebar for inputs
    st.sidebar.header("Race Configuration")
    
    # Tyre settings
    st.sidebar.subheader("Tyre Configuration")
    compound = st.sidebar.selectbox(
        "Tyre Compound",
        options=["SOFT", "MEDIUM", "HARD"],
        index=1
    )
    
    tyre_age = st.sidebar.slider(
        "Tyre Age (laps)",
        min_value=1,
        max_value=50,
        value=15,
        help="Number of laps completed on current tyres"
    )
    
    # Driving style
    st.sidebar.subheader("Driving Style")
    speed_profile = st.sidebar.selectbox(
        "Speed Profile",
        options=["Conservative", "Aggressive", "Consistent"],
        index=2
    )
    
    # Weather conditions
    st.sidebar.subheader("Weather Conditions")
    weather_temp = st.sidebar.slider(
        "Air Temperature (°C)",
        min_value=15.0,
        max_value=40.0,
        value=25.0,
        step=0.5
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Pace Delta Prediction")
        
        if model is not None:
            # Generate prediction
            sequence = create_sample_sequence(
                speed_profile, tyre_age, compound, weather_temp
            )
            
            # Make prediction
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                prediction = model(sequence_tensor).item()
            
            # Display prediction
            st.metric(
                label="Predicted Pace Delta",
                value=f"{prediction:.3f} seconds",
                delta=f"vs baseline lap time",
                help="Positive values indicate slower lap times, negative values indicate faster lap times"
            )
            
            # Interpretation
            if prediction > 1.0:
                interpretation = "⚠️ Significant pace loss expected"
                color = "red"
            elif prediction > 0.5:
                interpretation = "⚡ Moderate pace loss expected"
                color = "orange"
            elif prediction > -0.5:
                interpretation = "✅ Consistent pace expected"
                color = "green"
            else:
                interpretation = "🚀 Pace improvement expected"
                color = "blue"
            
            st.markdown(f"**Interpretation:** <span style='color: {color}'>{interpretation}</span>", 
                       unsafe_allow_html=True)
            
        else:
            st.warning("No trained model found. Please train a model first.")
            st.info("Available model paths checked:")
            model_paths = [
                "models/checkpoints/lightning_logs/version_0/checkpoints/last.ckpt",
                "models/checkpoints/f1-forecaster-best.ckpt",
                "cluster_results/checkpoints/last.ckpt"
            ]
            for path in model_paths:
                status = "✅" if Path(path).exists() else "❌"
                st.write(f"{status} {path}")
    
    with col2:
        st.header("Input Features")
        
        if model is not None:
            # Display current configuration
            st.write("**Current Configuration:**")
            st.write(f"• Compound: {compound}")
            st.write(f"• Tyre Age: {tyre_age} laps")
            st.write(f"• Speed Profile: {speed_profile}")
            st.write(f"• Air Temperature: {weather_temp}°C")
            
            # Show feature importance (simplified)
            st.subheader("Key Factors")
            factors = {
                "Tyre Age": min(tyre_age / 30, 1.0),
                "Compound Softness": {"SOFT": 0.9, "MEDIUM": 0.5, "HARD": 0.1}[compound],
                "Temperature": (weather_temp - 20) / 20,
                "Driving Aggression": {"Conservative": 0.2, "Consistent": 0.5, "Aggressive": 0.9}[speed_profile]
            }
            
            for factor, value in factors.items():
                st.progress(value, text=f"{factor}: {value:.1%}")
    
    # Tyre degradation analysis
    st.header("Tyre Degradation Analysis")
    
    if model is not None:
        # Generate predictions for different tyre ages
        tyre_ages = list(range(1, 51))
        predictions = []
        
        for age in tyre_ages:
            sequence = create_sample_sequence(
                speed_profile, age, compound, weather_temp
            )
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                pred = model(sequence_tensor).item()
                predictions.append(pred)
        
        # Create degradation plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tyre_ages,
            y=predictions,
            mode='lines+markers',
            name=f'{compound} compound',
            line=dict(width=3),
            marker=dict(size=6)
        ))
        
        # Highlight current tyre age
        fig.add_vline(
            x=tyre_age,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current: {tyre_age} laps"
        )
        
        fig.update_layout(
            title="Pace Delta vs Tyre Age",
            xaxis_title="Tyre Age (laps)",
            yaxis_title="Pace Delta (seconds)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Compound comparison
        st.subheader("Compound Comparison")
        
        compound_predictions = {}
        for comp in ["SOFT", "MEDIUM", "HARD"]:
            sequence = create_sample_sequence(
                speed_profile, tyre_age, comp, weather_temp
            )
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                pred = model(sequence_tensor).item()
                compound_predictions[comp] = pred
        
        # Bar chart for compound comparison
        fig_comp = go.Figure()
        colors = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white'}
        
        fig_comp.add_trace(go.Bar(
            x=list(compound_predictions.keys()),
            y=list(compound_predictions.values()),
            marker_color=[colors[comp] for comp in compound_predictions.keys()],
            marker_line_color='black',
            marker_line_width=2
        ))
        
        fig_comp.update_layout(
            title=f"Pace Delta by Compound (at {tyre_age} laps)",
            xaxis_title="Compound",
            yaxis_title="Pace Delta (seconds)",
            height=300
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # Model information
    with st.expander("Model Information"):
        if model is not None:
            st.write(f"**Model Path:** {model_path}")
            st.write(f"**Model Type:** Transformer-based Sequence-to-Sequence")
            st.write(f"**Input Features:** {data['features'].shape[-1]}")
            st.write(f"**Sequence Length:** {data['features'].shape[1]}")
            st.write(f"**Training Data:** {len(data['features'])} samples from {data['year']}")
        
        st.write("**Feature Description:**")
        st.write("• Speed: Mean, max, min speed per micro-sector")
        st.write("• Throttle: Throttle application patterns")
        st.write("• Braking: Brake usage and intensity")
        st.write("• Track Position: Micro-sector progression")
        st.write("• Tyre: Compound type and age")
        st.write("• Weather: Air temperature, track temperature, humidity")
    
    # Dataset statistics
    with st.expander("Dataset Statistics"):
        if data is not None:
            st.write(f"**Dataset Year:** {data['year']}")
            st.write(f"**Races Included:** {len(data['races'])}")
            st.write(f"**Total Samples:** {len(data['features'])}")
            st.write(f"**Feature Dimensions:** {data['features'].shape}")
            
            # Show pace delta distribution
            fig_hist = px.histogram(
                x=data['targets'],
                nbins=50,
                title="Distribution of Pace Deltas in Training Data"
            )
            fig_hist.update_xaxes(title_text="Pace Delta (seconds)")
            fig_hist.update_yaxes(title_text="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()
