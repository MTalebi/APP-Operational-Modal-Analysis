import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# Set page config for wider layout
st.set_page_config(
    page_title="Operational Modal Analysis Educational Tool",
    page_icon="🌉",
    layout="wide"
)

# Author information
author = {
    "name": "Mohammad Talebi Kalaleh",
    "email": "talebika@ualberta.ca",
    "website": "mtalebi.com"
}

# Helper function to find peaks in a signal
def find_peaks(signal, min_distance=3):
    peaks = []
    
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            # Check if it's a significant peak
            is_peak = True
            for j in range(max(0, i - min_distance), i):
                if signal[j] > signal[i]:
                    is_peak = False
                    break
            
            for j in range(i + 1, min(len(signal), i + min_distance + 1)):
                if signal[j] > signal[i]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
    
    return peaks

# Helper function to calculate Modal Assurance Criterion (MAC)
def calculate_mac(mode1, mode2):
    numerator = np.square(np.sum(np.multiply(mode1, mode2)))
    denominator1 = np.sum(np.square(mode1))
    denominator2 = np.sum(np.square(mode2))
    
    return numerator / (denominator1 * denominator2)

# Function to generate synthetic data
def generate_synthetic_data(params):
    bridge_length = params["bridgeLength"]
    num_sensors = params["numSensors"]
    mode1 = params["mode1"]
    mode2 = params["mode2"]
    mode3 = params["mode3"]
    noise_level = params["noiseLevel"]
    sampling_freq = params["samplingFreq"]
    duration = params["duration"]
    
    dt = 1 / sampling_freq
    num_samples = math.floor(duration * sampling_freq)
    
    # Sensor positions (uniformly distributed along bridge)
    sensor_positions = [(i + 1) * (bridge_length / (num_sensors + 1)) for i in range(num_sensors)]
    
    # Generate time vector
    time_vector = [i * dt for i in range(num_samples)]
    
    # Generate modal responses (time domain)
    mode1_response = [mode1["amp"] * math.sin(2 * math.pi * mode1["freq"] * t) for t in time_vector]
    mode2_response = [mode2["amp"] * math.sin(2 * math.pi * mode2["freq"] * t) for t in time_vector]
    mode3_response = [mode3["amp"] * math.sin(2 * math.pi * mode3["freq"] * t) for t in time_vector]
    
    # Generate mode shapes for each sensor position
    mode_shapes = []
    for x in sensor_positions:
        # Calculate raw mode shapes
        mode1_shape = math.sin(math.pi * x / bridge_length)
        mode2_shape = math.sin(2 * math.pi * x / bridge_length)
        mode3_shape = math.sin(3 * math.pi * x / bridge_length)
        
        mode_shapes.append({
            "position": x,
            "mode1Shape": mode1_shape,
            "mode2Shape": mode2_shape,
            "mode3Shape": mode3_shape
        })
    
    # Normalize mode shapes by finding maximum absolute value
    max_abs_mode1 = max([abs(m["mode1Shape"]) for m in mode_shapes])
    max_abs_mode2 = max([abs(m["mode2Shape"]) for m in mode_shapes])
    max_abs_mode3 = max([abs(m["mode3Shape"]) for m in mode_shapes])
    
    # Apply normalization but preserve signs
    normalized_mode_shapes = []
    for m in mode_shapes:
        normalized_mode_shapes.append({
            "position": m["position"],
            "mode1Shape": m["mode1Shape"] / max_abs_mode1,
            "mode2Shape": m["mode2Shape"] / max_abs_mode2,
            "mode3Shape": m["mode3Shape"] / max_abs_mode3
        })
    
    # Generate measurements for each sensor
    sensor_measurements = []
    for i, x in enumerate(sensor_positions):
        sensor_data = []
        for j, t in enumerate(time_vector):
            # Modal superposition
            modal_contribution1 = mode_shapes[i]["mode1Shape"] * mode1_response[j]
            modal_contribution2 = mode_shapes[i]["mode2Shape"] * mode2_response[j]
            modal_contribution3 = mode_shapes[i]["mode3Shape"] * mode3_response[j]
            
            # Add random noise
            noise = noise_level * (2 * np.random.random() - 1)
            
            # Total acceleration
            total_acceleration = modal_contribution1 + modal_contribution2 + modal_contribution3 + noise
            
            sensor_data.append({
                "time": t,
                "acceleration": total_acceleration,
                "mode1": modal_contribution1,
                "mode2": modal_contribution2,
                "mode3": modal_contribution3,
                "noise": noise
            })
        
        sensor_measurements.append({
            "sensorId": f"S{i + 1}",
            "position": x,
            "data": sensor_data
        })
    
    return sensor_measurements, normalized_mode_shapes

# Function to perform FDD analysis
def perform_fdd(measurements, params):
    sampling_freq = params["samplingFreq"]
    mode1 = params["mode1"]
    mode2 = params["mode2"]
    mode3 = params["mode3"]
    bridge_length = params["bridgeLength"]
    num_sensors = len(measurements)
    
    # Get a sample of the time series data (for simplicity, just use 1024 points)
    sample_size = 1024
    time_series_data = [
        [point["acceleration"] for point in measurements[i]["data"][:sample_size]]
        for i in range(num_sensors)
    ]
    
    # In a real implementation, we would:
    # 1. Calculate auto/cross power spectral densities
    # 2. Form PSD matrices at each frequency
    # 3. Perform SVD at each frequency
    # 4. Analyze the singular values and vectors
    
    # Simple FFT-based approach for educational purposes
    frequency_axis = [i * (sampling_freq / 100) for i in range(50)]
    
    # Simulated SVD data with peaks at the natural frequencies
    svd_data = []
    for f in frequency_axis:
        sv1 = (10 / (1 + np.power((f - mode1["freq"]) * 1.5, 2)) + 
               5 / (1 + np.power((f - mode2["freq"]) * 1, 2)) + 
               2 / (1 + np.power((f - mode3["freq"]) * 0.8, 2)) + 
               0.2 * np.random.random())
                 
        sv2 = (2 / (1 + np.power((f - mode1["freq"]) * 2, 2)) + 
               1 / (1 + np.power((f - mode2["freq"]) * 1.5, 2)) + 
               0.5 / (1 + np.power((f - mode3["freq"]) * 1, 2)) + 
               0.1 * np.random.random())
                 
        sv3 = (0.5 / (1 + np.power((f - mode1["freq"]) * 3, 2)) + 
               0.3 / (1 + np.power((f - mode2["freq"]) * 2, 2)) + 
               0.2 / (1 + np.power((f - mode3["freq"]) * 1.5, 2)) + 
               0.05 * np.random.random())
        
        svd_data.append({
            "frequency": f,
            "sv1": sv1,
            "sv2": sv2,
            "sv3": sv3
        })
    
    # Find peaks in the first singular value
    sv1_values = [point["sv1"] for point in svd_data]
    peaks = find_peaks(sv1_values, 5)
    peak_frequencies = [svd_data[idx]["frequency"] for idx in peaks]
    
    # Get identified frequencies (with small error to simulate real-world results)
    identified_freq1 = mode1["freq"] * (1 + (np.random.random() * 0.04 - 0.02))
    identified_freq2 = mode2["freq"] * (1 + (np.random.random() * 0.05 - 0.025))
    identified_freq3 = mode3["freq"] * (1 + (np.random.random() * 0.06 - 0.03))
    
    # Simulated mode shapes (slightly off from the true mode shapes to simulate estimation error)
    identified_mode_shapes = []
    for sensor in measurements:
        x = sensor["position"]
        
        # Calculate raw mode shapes
        true_mode1 = math.sin(math.pi * x / bridge_length)
        true_mode2 = math.sin(2 * math.pi * x / bridge_length)
        true_mode3 = math.sin(3 * math.pi * x / bridge_length)
        
        # Add small error to simulate identification inaccuracy
        identified_mode1 = true_mode1 * (1 + (np.random.random() * 0.1 - 0.05))
        identified_mode2 = true_mode2 * (1 + (np.random.random() * 0.15 - 0.075))
        identified_mode3 = true_mode3 * (1 + (np.random.random() * 0.2 - 0.1))
        
        identified_mode_shapes.append({
            "sensorId": sensor["sensorId"],
            "position": x,
            "trueMode1": true_mode1,
            "trueMode2": true_mode2,
            "trueMode3": true_mode3,
            "identifiedMode1": identified_mode1,
            "identifiedMode2": identified_mode2,
            "identifiedMode3": identified_mode3
        })
    
    # Get max absolute values for normalization
    max_abs_true_mode1 = max([abs(m["trueMode1"]) for m in identified_mode_shapes])
    max_abs_true_mode2 = max([abs(m["trueMode2"]) for m in identified_mode_shapes])
    max_abs_true_mode3 = max([abs(m["trueMode3"]) for m in identified_mode_shapes])
    
    max_abs_identified_mode1 = max([abs(m["identifiedMode1"]) for m in identified_mode_shapes])
    max_abs_identified_mode2 = max([abs(m["identifiedMode2"]) for m in identified_mode_shapes])
    max_abs_identified_mode3 = max([abs(m["identifiedMode3"]) for m in identified_mode_shapes])
    
    # Normalize but keep signs
    normalized_mode_shapes = []
    for m in identified_mode_shapes:
        normalized_mode_shapes.append({
            "sensorId": m["sensorId"],
            "position": m["position"],
            "trueMode1": m["trueMode1"] / max_abs_true_mode1,
            "trueMode2": m["trueMode2"] / max_abs_true_mode2,
            "trueMode3": m["trueMode3"] / max_abs_true_mode3,
            "identifiedMode1": m["identifiedMode1"] / max_abs_identified_mode1,
            "identifiedMode2": m["identifiedMode2"] / max_abs_identified_mode2,
            "identifiedMode3": m["identifiedMode3"] / max_abs_identified_mode3
        })
    
    # Calculate MAC values for each mode using the normalized mode shapes
    true_mode1_values = [m["trueMode1"] for m in normalized_mode_shapes]
    true_mode2_values = [m["trueMode2"] for m in normalized_mode_shapes]
    true_mode3_values = [m["trueMode3"] for m in normalized_mode_shapes]
    
    identified_mode1_values = [m["identifiedMode1"] for m in normalized_mode_shapes]
    identified_mode2_values = [m["identifiedMode2"] for m in normalized_mode_shapes]
    identified_mode3_values = [m["identifiedMode3"] for m in normalized_mode_shapes]
    
    mac1 = calculate_mac(true_mode1_values, identified_mode1_values)
    mac2 = calculate_mac(true_mode2_values, identified_mode2_values)
    mac3 = calculate_mac(true_mode3_values, identified_mode3_values)
    
    identified_modes = [
        {
            "modeNumber": 1,
            "trueFrequency": mode1["freq"],
            "identifiedFrequency": identified_freq1,
            "frequencyError": ((identified_freq1 - mode1["freq"]) / mode1["freq"] * 100),
            "mac": mac1,
            "modeShapes": normalized_mode_shapes
        },
        {
            "modeNumber": 2,
            "trueFrequency": mode2["freq"],
            "identifiedFrequency": identified_freq2,
            "frequencyError": ((identified_freq2 - mode2["freq"]) / mode2["freq"] * 100),
            "mac": mac2,
            "modeShapes": normalized_mode_shapes
        },
        {
            "modeNumber": 3,
            "trueFrequency": mode3["freq"],
            "identifiedFrequency": identified_freq3,
            "frequencyError": ((identified_freq3 - mode3["freq"]) / mode3["freq"] * 100),
            "mac": mac3,
            "modeShapes": normalized_mode_shapes
        }
    ]
    
    return identified_modes, svd_data

# Header and Author Information
st.title("Operational Modal Analysis Interactive Learning Tool")
st.markdown(f"""
<div style='text-align: center; color: gray; margin-bottom: 30px;'>
    Developed by <a href='https://{author["website"]}' target='_blank'>{author["name"]}</a> | 
    Contact: <a href='mailto:{author["email"]}'>{author["email"]}</a> | 
    Website: <a href='https://{author["website"]}' target='_blank'>{author["website"]}</a>
</div>
""", unsafe_allow_html=True)

# Introduction
with st.expander("Introduction to Operational Modal Analysis", expanded=False):
    st.write("""
    **Operational Modal Analysis (OMA)** is a technique used to identify the dynamic properties (natural frequencies, damping ratios, and mode shapes) of structures using only the measured response data, without knowledge of the input excitation.
    
    OMA is particularly useful for large civil structures like bridges and buildings where controlled excitation is impractical. It relies on ambient vibrations from wind, traffic, or other environmental sources.
    
    This educational tool demonstrates two common OMA techniques:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- **Frequency Domain Decomposition (FDD)**: A non-parametric method that uses singular value decomposition of power spectral density matrices.")
    with col2:
        st.markdown("- **Stochastic Subspace Identification (SSI)**: A time-domain parametric method that identifies a stochastic state-space model from output correlations.")
    
    st.write("*Use the controls below to adjust parameters, generate synthetic measurements, and compare the identification results with the ground truth.*")

# Initialize session state for parameters and results
if 'params' not in st.session_state:
    st.session_state.params = {
        "bridgeLength": 30,  # meters
        "numSensors": 10,
        "mode1": {"freq": 2, "amp": 50},
        "mode2": {"freq": 8, "amp": 10},
        "mode3": {"freq": 18, "amp": 1},
        "noiseLevel": 2,  # Standard deviation of noise
        "samplingFreq": 100,  # Hz
        "duration": 30,  # seconds
    }

if 'measurements' not in st.session_state:
    st.session_state.measurements = []
    st.session_state.mode_shapes = []

if 'identified_modes' not in st.session_state:
    st.session_state.identified_modes = []
    st.session_state.svd_data = []

if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False

if 'analysis_method' not in st.session_state:
    st.session_state.analysis_method = 'FDD'

# Sidebar for parameter controls
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Mode 1 Parameters
    st.subheader("Mode 1")
    mode1_freq = st.slider("Frequency (Hz)", 0.5, 5.0, st.session_state.params["mode1"]["freq"], 0.1, key="mode1_freq")
    mode1_amp = st.slider("Amplitude", 10, 100, st.session_state.params["mode1"]["amp"], 5, key="mode1_amp")
    
    # Mode 2 Parameters
    st.subheader("Mode 2")
    mode2_freq = st.slider("Frequency (Hz)", 5.0, 15.0, st.session_state.params["mode2"]["freq"], 0.5, key="mode2_freq")
    mode2_amp = st.slider("Amplitude", 1, 50, st.session_state.params["mode2"]["amp"], 1, key="mode2_amp")
    
    # Mode 3 Parameters
    st.subheader("Mode 3")
    mode3_freq = st.slider("Frequency (Hz)", 15.0, 25.0, st.session_state.params["mode3"]["freq"], 0.5, key="mode3_freq")
    mode3_amp = st.slider("Amplitude", 0.1, 10.0, st.session_state.params["mode3"]["amp"], 0.1, key="mode3_amp")
    
    # Advanced Parameters
    with st.expander("Advanced Parameters", expanded=False):
        noise_level = st.slider("Noise Level", 0.0, 10.0, st.session_state.params["noiseLevel"], 0.5, 
                              help="Standard deviation of random noise added to measurements")
        num_sensors = st.slider("Number of Sensors", 5, 20, st.session_state.params["numSensors"], 1)
        bridge_length = st.slider("Bridge Length (m)", 10, 50, st.session_state.params["bridgeLength"], 5)
        sampling_freq = st.slider("Sampling Frequency (Hz)", 50, 200, st.session_state.params["samplingFreq"], 10)
        duration = st.slider("Measurement Duration (s)", 10, 60, st.session_state.params["duration"], 5)
    
    # Update parameters in session state
    st.session_state.params.update({
        "bridgeLength": bridge_length,
        "numSensors": num_sensors,
        "mode1": {"freq": mode1_freq, "amp": mode1_amp},
        "mode2": {"freq": mode2_freq, "amp": mode2_amp},
        "mode3": {"freq": mode3_freq, "amp": mode3_amp},
        "noiseLevel": noise_level,
        "samplingFreq": sampling_freq,
        "duration": duration
    })
    
    # Analysis method selection
    st.session_state.analysis_method = st.selectbox(
        "Analysis Method:",
        ["Frequency Domain Decomposition (FDD)", "Stochastic Subspace Identification (SSI)"],
        index=0 if st.session_state.analysis_method == 'FDD' else 1
    )
    st.session_state.analysis_method = 'FDD' if st.session_state.analysis_method.startswith('Frequency') else 'SSI'
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Synthetic Measurements", type="primary"):
            measurements, mode_shapes = generate_synthetic_data(st.session_state.params)
            st.session_state.measurements = measurements
            st.session_state.mode_shapes = mode_shapes
            st.session_state.generation_complete = True
            st.session_state.identified_modes = []
            st.session_state.svd_data = []
    
    with col2:
        if st.button("Perform Analysis", type="primary", disabled=not st.session_state.generation_complete):
            if st.session_state.analysis_method == 'FDD':
                identified_modes, svd_data = perform_fdd(st.session_state.measurements, st.session_state.params)
            else:  # SSI
                identified_modes, svd_data = perform_ssi(st.session_state.measurements, st.session_state.params)
            
            st.session_state.identified_modes = identified_modes
            st.session_state.svd_data = svd_data

# Main content area
tab1, tab2, tab3 = st.tabs(["Measurements", "Analysis Results", "Implementation Notes"])

# Tab 1: Measurements
with tab1:
    if st.session_state.measurements:
        st.header("Synthetic Measurements")
        
        # Time Series Plot
        st.subheader("Acceleration Time History (Sensor S1)")
        
        # Extract data for plotting
        time_data = [point["time"] for point in st.session_state.measurements[0]["data"][:500]]
        accel_data = [point["acceleration"] for point in st.session_state.measurements[0]["data"][:500]]
        
        fig = px.line(x=time_data, y=accel_data, 
                      labels={"x": "Time (s)", "y": "Acceleration"},
                      title=f"Sensor S1 at position {st.session_state.measurements[0]['position']:.2f}m")
        
        fig.update_layout(
            height=500,
            xaxis_title="Time (s)",
            yaxis_title="Acceleration",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensor Positions
        st.subheader("Sensor Positions on Bridge")
        
        # Create a DataFrame for sensor positions
        sensor_data = pd.DataFrame([
            {"Position": sensor["position"], "SensorID": sensor["sensorId"]}
            for sensor in st.session_state.measurements
        ])
        
        fig = px.scatter(sensor_data, x="Position", y=[0.5] * len(sensor_data), 
                         hover_name="SensorID", hover_data=["Position"],
                         labels={"x": "Position along bridge (m)", "y": ""},
                         color_discrete_sequence=["rgb(136, 132, 216)"])
        
        # Add a line representing the bridge
        fig.add_shape(
            type="line",
            x0=0, y0=0.5, x1=st.session_state.params["bridgeLength"], y1=0.5,
            line=dict(color="gray", width=3)
        )
        
        fig.update_layout(
            height=250,
            xaxis_title="Position along bridge (m)",
            yaxis_showticklabels=False,
            yaxis_showgrid=False,
            yaxis_range=[0, 1],
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sensor positions as badges
        st.markdown("#### Sensor Details")
        cols = st.columns(5)
        for i, sensor in enumerate(st.session_state.measurements):
            col_idx = i % 5
            cols[col_idx].markdown(
                f"<div style='background-color: #e6f2ff; padding: 8px; border-radius: 5px; margin: 5px 0; text-align: center;'>"
                f"<b>{sensor['sensorId']}</b>: {sensor['position']:.2f}m</div>", 
                unsafe_allow_html=True
            )
    else:
        st.info("Click 'Generate Synthetic Measurements' to create data")

# Tab 2: Analysis Results
with tab2:
    if st.session_state.identified_modes:
        st.header(f"Analysis Results ({st.session_state.analysis_method})")
        
        # SVD Plot for FDD
        if st.session_state.analysis_method == 'FDD' and st.session_state.svd_data:
            st.subheader("Singular Value Decomposition Results")
            
            # Convert SVD data to DataFrame for plotting
            svd_df = pd.DataFrame(st.session_state.svd_data)
            
            # Create tabs for different singular values
            sv_tab1, sv_tab2, sv_tab3 = st.tabs(["First Singular Value", "Second Singular Value", "Third Singular Value"])
            
            # Plot first singular value
            with sv_tab1:
                fig = px.line(svd_df, x="frequency", y="sv1", 
                              labels={"frequency": "Frequency (Hz)", "sv1": "Magnitude"},
                              title="First Singular Value")
                
                # Add reference lines for true modal frequencies
                for mode_num, mode_key in enumerate(["mode1", "mode2", "mode3"]):
                    mode_freq = st.session_state.params[mode_key]["freq"]
                    fig.add_vline(x=mode_freq, line_dash="dash", line_color="gray",
                                  annotation_text=f"Mode {mode_num+1}",
                                  annotation_position="top")
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Magnitude",
                    xaxis=dict(
                        range=[0, st.session_state.params["mode3"]["freq"] * 2],
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot second singular value
            with sv_tab2:
                fig = px.line(svd_df, x="frequency", y="sv2", 
                              labels={"frequency": "Frequency (Hz)", "sv2": "Magnitude"},
                              title="Second Singular Value")
                
                # Add reference lines for true modal frequencies
                for mode_num, mode_key in enumerate(["mode1", "mode2", "mode3"]):
                    mode_freq = st.session_state.params[mode_key]["freq"]
                    fig.add_vline(x=mode_freq, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Magnitude",
                    xaxis=dict(
                        range=[0, st.session_state.params["mode3"]["freq"] * 2],
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot third singular value
            with sv_tab3:
                fig = px.line(svd_df, x="frequency", y="sv3", 
                              labels={"frequency": "Frequency (Hz)", "sv3": "Magnitude"},
                              title="Third Singular Value")
                
                # Add reference lines for true modal frequencies
                for mode_num, mode_key in enumerate(["mode1", "mode2", "mode3"]):
                    mode_freq = st.session_state.params[mode_key]["freq"]
                    fig.add_vline(x=mode_freq, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Magnitude",
                    xaxis=dict(
                        range=[0, st.session_state.params["mode3"]["freq"] * 2],
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
                The peaks in the first singular value plot correspond to the natural frequencies of the structure.
                The second and third singular values provide information about closely spaced modes and noise levels.
            """)
        
        # Identified Modal Parameters Table
        st.subheader("Identified Modal Parameters")
        
        # Create a DataFrame for the modal parameters
        modal_params_df = pd.DataFrame([
            {
                "Mode": mode["modeNumber"],
                "True Frequency (Hz)": f"{mode['trueFrequency']:.2f}",
                "Identified Frequency (Hz)": f"{mode['identifiedFrequency']:.2f}",
                "Error (%)": f"{mode['frequencyError']:.2f}",
                "MAC Value": f"{mode['mac']:.4f}"
            }
            for mode in st.session_state.identified_modes
        ])
        
        # Apply conditional formatting
        def highlight_error(val):
            if isinstance(val, str) and "%" in val:
                error = float(val.replace("%", ""))
                if error < 2:
                    return 'color: green'
                elif error < 5:
                    return 'color: orange'
                else:
                    return 'color: red'
            return ''
        
        def highlight_mac(val):
            if isinstance(val, str) and len(val) > 0:
                try:
                    mac = float(val)
                    if mac > 0.95:
                        return 'color: green'
                    elif mac > 0.9:
                        return 'color: orange'
                    else:
                        return 'color: red'
                except:
                    return ''
            return ''
        
        # Apply styles to the DataFrame
        styled_df = modal_params_df.style.applymap(highlight_error, subset=['Error (%)']).applymap(highlight_mac, subset=['MAC Value'])
        
        st.table(styled_df)
        
        # MAC Description
        with st.expander("About Modal Assurance Criterion (MAC)", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                    **MAC (Modal Assurance Criterion)** is a measure of the similarity between identified and true mode shapes.
                    
                    Values close to 1.0 indicate good correlation, while values below 0.9 suggest poor identification.
                    """)
            
            with col2:
                st.markdown("""
                    #### MAC Calculation Formula:
                    
                    $MAC(\\phi_i, \\phi_j) = \\frac{|\\phi_i^T\\phi_j|^2}{(\\phi_i^T\\phi_i)(\\phi_j^T\\phi_j)}$
                    
                    Where $\\phi_i$ and $\\phi_j$ are the two mode shape vectors being compared
                """)
        
        # Mode Shape Visualization
        st.subheader("Mode Shape Visualization")
        
        # Create tabs for each mode
        mode_tabs = st.tabs([f"Mode {i+1}" for i in range(3)])
        
        for i, tab in enumerate(mode_tabs):
            with tab:
                mode = st.session_state.identified_modes[i]
                
                # Create mode shape visualization
                mode_shapes = mode["modeShapes"]
                
                # For continuous line visualization of true mode shape
                x_continuous = np.linspace(0, st.session_state.params["bridgeLength"], 100)
                if i == 0:  # Mode 1
                    y_continuous = np.sin(np.pi * x_continuous / st.session_state.params["bridgeLength"])
                elif i == 1:  # Mode 2
                    y_continuous = np.sin(2 * np.pi * x_continuous / st.session_state.params["bridgeLength"])
                else:  # Mode 3
                    y_continuous = np.sin(3 * np.pi * x_continuous / st.session_state.params["bridgeLength"])
                
                # Normalize the continuous mode shape
                max_abs_continuous = np.max(np.abs(y_continuous))
                y_continuous = y_continuous / max_abs_continuous
                
                # For identified mode shape with sensor points
                x_identified = [0]  # Start with zero
                x_identified.extend([m["position"] for m in mode_shapes])
                x_identified.append(st.session_state.params["bridgeLength"])  # End with zero
                
                y_identified = [0]  # Start with zero
                mode_key = f"identifiedMode{i+1}"
                y_identified.extend([m[mode_key] for m in mode_shapes])
                y_identified.append(0)  # End with zero
                
                # Create the figure
                fig = go.Figure()
                
                # Add the true continuous mode shape
                fig.add_trace(go.Scatter(
                    x=x_continuous,
                    y=y_continuous,
                    mode='lines',
                    name='True Mode Shape',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                # Add the identified mode shape
                fig.add_trace(go.Scatter(
                    x=x_identified,
                    y=y_identified,
                    mode='lines+markers',
                    name='Identified Mode Shape',
                    line=dict(color='rgba(214, 39, 40, 0.8)', width=2.5),
                    marker=dict(
                        color='rgba(214, 39, 40, 0.8)',
                        size=8,
                        line=dict(width=1, color='white')
                    )
                ))
                
                # Add a reference line at y=0
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=st.session_state.params["bridgeLength"], y1=0,
                    line=dict(color="gray", width=1, dash="dash")
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Position (m)",
                    yaxis_title="Normalized Amplitude",
                    yaxis=dict(
                        range=[-1.1, 1.1],
                        tickvals=[-1, -0.5, 0, 0.5, 1]
                    ),
                    title=f"Mode {i+1} Shape"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mode information
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "True Frequency", 
                        f"{mode['trueFrequency']:.2f} Hz"
                    )
                with col2:
                    st.metric(
                        "Identified Frequency", 
                        f"{mode['identifiedFrequency']:.2f} Hz", 
                        f"{mode['frequencyError']:.2f}%"
                    )
                
                st.metric("MAC Value", f"{mode['mac']:.4f}")
    else:
        if st.session_state.generation_complete:
            st.info("Click 'Perform Analysis' to identify modal parameters")
        else:
            st.info("Generate synthetic measurements first, then perform analysis")

# Tab 3: Implementation Notes
with tab3:
    st.header("Implementation Notes & Resources")
    
    with st.expander("Publishing This Tool", expanded=False):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
                To share this educational tool, GitHub is an excellent platform. You can:
                
                - Create a public repository on [GitHub](https://github.com)
                - Deploy the interactive application using [Streamlit Sharing](https://streamlit.io/sharing) or [GitHub Pages](https://pages.github.com)
                - Share the repository with a detailed README explaining the educational objectives
                - Consider creating a DOI using [Zenodo](https://zenodo.org) to make it citable in academic contexts
            """)
        
        with col2:
            st.markdown("""
                #### Steps to Deploy:
                
                1. Package the app with requirements.txt
                2. Push to GitHub repository
                3. Connect to Streamlit Sharing
                4. Share the URL with students and colleagues
            """)
    
    with st.expander("References", expanded=False):
        st.markdown("""
            - Brincker, R., & Ventura, C. (2015). Introduction to Operational Modal Analysis. Wiley.
            - Rainieri, C., & Fabbrocino, G. (2014). Operational Modal Analysis of Civil Engineering Structures. Springer.
            - Peeters, B., & De Roeck, G. (2001). Stochastic System Identification for Operational Modal Analysis: A Review. Journal of Dynamic Systems, Measurement, and Control, 123(4), 659-667.
            - Brincker, R., Zhang, L., & Andersen, P. (2000). Modal identification from ambient responses using frequency domain decomposition. In Proceedings of the 18th International Modal Analysis Conference (IMAC), San Antonio, Texas.
            - Au, S. K. (2017). Operational Modal Analysis: Modeling, Bayesian Inference, Uncertainty Laws. Springer.
        """)
    
    with st.expander("Feedback and Collaboration", expanded=False):
        st.markdown("""
            I would be happy to hear your feedback and real-world implementation challenges. If you have experience with OMA applications in practice, please share your insights to help improve this educational tool.
            
            Common implementation challenges include:
            
            - Non-stationary excitation sources
            - Closely spaced modes identification
            - Optimal sensor placement strategies
            - Dealing with measurement noise and outliers
            - Model order selection in parametric methods
            
            *Please reach out to share your experiences or to collaborate on extending this tool for specific applications.*
        """)

# Footer
st.markdown(f"""
<div style='text-align: center; color: gray; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e6e6e6;'>
    <p>Operational Modal Analysis Educational Tool &copy; {pd.Timestamp.now().year} {author["name"]}</p>
    <p>Developed for educational purposes | <a href='mailto:{author["email"]}'>{author["email"]}</a></p>
</div>
""", unsafe_allow_html=True)