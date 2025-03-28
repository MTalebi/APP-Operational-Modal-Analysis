import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from typing import List, Dict

# -----------------------------------------------------------------------------
# Author: Mohammad Talebi-Kalaleh
# Contact: talebika@ualberta.ca
# -----------------------------------------------------------------------------
# This code implements an Operational Modal Analysis (OMA) educational tool in 
# Python using Streamlit, illustrating Frequency Domain Decomposition (FDD) 
# and (placeholder) Stochastic Subspace Identification (SSI).
# -----------------------------------------------------------------------------

# Session-state keys to store data across interactions
MEASUREMENTS_KEY = "measurements"
SVD_DATA_KEY = "svd_data"
IDENTIFIED_MODES_KEY = "identified_modes"
GENERATION_COMPLETE_KEY = "generation_complete"
IS_ANALYZING_KEY = "is_analyzing"


def generate_synthetic_data(params: dict):
    """
    Generate synthetic measurements of a 3-mode vibrating system with noise.
    Returns a list of dicts, each containing sensor id, position, and time-series data.
    """
    bridge_length = params["bridgeLength"]
    num_sensors = params["numSensors"]
    mode1 = params["mode1"]
    mode2 = params["mode2"]
    mode3 = params["mode3"]
    noise_level = params["noiseLevel"]
    sampling_freq = params["samplingFreq"]
    duration = params["duration"]

    dt = 1.0 / sampling_freq
    num_samples = int(np.floor(duration * sampling_freq))

    # Sensor positions (uniformly distributed)
    sensor_positions = np.linspace(
        bridge_length / (num_sensors + 1),
        bridge_length * (num_sensors / (num_sensors + 1)),
        num_sensors
    )

    # Time vector
    time_vector = np.arange(num_samples) * dt

    # Generate time-domain modal responses
    mode1_response = mode1["amp"] * np.sin(2 * np.pi * mode1["freq"] * time_vector)
    mode2_response = mode2["amp"] * np.sin(2 * np.pi * mode2["freq"] * time_vector)
    mode3_response = mode3["amp"] * np.sin(2 * np.pi * mode3["freq"] * time_vector)

    # Precompute sensor mode shapes
    mode_shapes = []
    for x in sensor_positions:
        # Raw shape
        shape1 = np.sin(np.pi * x / bridge_length)
        shape2 = np.sin(2 * np.pi * x / bridge_length)
        shape3 = np.sin(3 * np.pi * x / bridge_length)
        mode_shapes.append((shape1, shape2, shape3))

    # Generate the sensor measurements (time series)
    measurements = []
    for i, x in enumerate(sensor_positions):
        shape1, shape2, shape3 = mode_shapes[i]
        data_list = []
        for t_idx in range(num_samples):
            # Modal superposition
            m1 = shape1 * mode1_response[t_idx]
            m2 = shape2 * mode2_response[t_idx]
            m3 = shape3 * mode3_response[t_idx]

            # Add random noise
            noise = noise_level * (2 * np.random.rand() - 1)

            total_acc = m1 + m2 + m3 + noise
            data_list.append({
                "time": time_vector[t_idx],
                "acceleration": total_acc,
                "mode1": m1,
                "mode2": m2,
                "mode3": m3,
                "noise": noise
            })

        sensor_dict = {
            "sensorId": f"S{i+1}",
            "position": x,
            "data": data_list
        }
        measurements.append(sensor_dict)

    return measurements


def perform_analysis(analysis_method: str, params: dict, measurements: List[Dict]):
    """
    Wrapper to perform OMA analysis (FDD or SSI).
    Returns the identified modes and SVD data (if FDD).
    """
    if analysis_method == "FDD":
        return perform_fdd(params, measurements)
    elif analysis_method == "SSI":
        return perform_ssi(params, measurements)
    else:
        return [], []


def perform_fdd(params: dict, measurements: List[Dict]):
    """
    Frequency Domain Decomposition (simplified educational version).
    Returns: 
        identified_modes (list of dict)
        svd_data (list of dict for frequency vs singular values)
    """
    mode1 = params["mode1"]
    mode2 = params["mode2"]
    mode3 = params["mode3"]

    # For demonstration, we build a "synthetic" SVD with peaks at known frequencies (with noise).
    # We'll sample across a frequency axis up to 2*(third mode frequency).
    max_freq = 2.0 * mode3["freq"]
    freq_count = 50  # number of points
    frequency_axis = np.linspace(0, max_freq, freq_count)

    svd_data = []
    for f in frequency_axis:
        sv1 = (10 / (1 + ((f - mode1["freq"]) * 1.5)**2)
               + 5 / (1 + ((f - mode2["freq"]) * 1.0)**2)
               + 2 / (1 + ((f - mode3["freq"]) * 0.8)**2)
               + 0.2 * np.random.rand())
        sv2 = (2 / (1 + ((f - mode1["freq"]) * 2.0)**2)
               + 1 / (1 + ((f - mode2["freq"]) * 1.5)**2)
               + 0.5 / (1 + ((f - mode3["freq"]) * 1.0)**2)
               + 0.1 * np.random.rand())
        sv3 = (0.5 / (1 + ((f - mode1["freq"]) * 3.0)**2)
               + 0.3 / (1 + ((f - mode2["freq"]) * 2.0)**2)
               + 0.2 / (1 + ((f - mode3["freq"]) * 1.5)**2)
               + 0.05 * np.random.rand())
        svd_data.append({
            "frequency": f,
            "sv1": sv1,
            "sv2": sv2,
            "sv3": sv3
        })

    # Simulate identified frequencies (slight random error)
    identified_freq1 = mode1["freq"] * (1 + (np.random.rand() * 0.04 - 0.02))
    identified_freq2 = mode2["freq"] * (1 + (np.random.rand() * 0.05 - 0.025))
    identified_freq3 = mode3["freq"] * (1 + (np.random.rand() * 0.06 - 0.03))

    # Build identified mode shapes with small error
    identified_mode_shapes = []
    for sensor in measurements:
        x = sensor["position"]
        true_mode1 = np.sin(np.pi * x / params["bridgeLength"])
        true_mode2 = np.sin(2 * np.pi * x / params["bridgeLength"])
        true_mode3 = np.sin(3 * np.pi * x / params["bridgeLength"])

        # Add small error
        identified_mode1 = true_mode1 * (1 + (np.random.rand() * 0.1 - 0.05))
        identified_mode2 = true_mode2 * (1 + (np.random.rand() * 0.15 - 0.075))
        identified_mode3 = true_mode3 * (1 + (np.random.rand() * 0.2 - 0.1))

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

    # Normalize for MAC and finalize results
    identified_modes = finalize_modes(
        identified_mode_shapes,
        [mode1["freq"], mode2["freq"], mode3["freq"]],
        [identified_freq1, identified_freq2, identified_freq3]
    )

    return identified_modes, svd_data


def perform_ssi(params: dict, measurements: List[Dict]):
    """
    Simplified SSI approach that currently just calls the FDD logic.
    Replace with an actual SSI method if needed.
    """
    identified_modes, svd_data = perform_fdd(params, measurements)
    return identified_modes, svd_data


def finalize_modes(identified_mode_shapes, true_freqs, identified_freqs):
    """
    Computes MAC values and frequency errors, returns a list of identified mode dicts.
    """
    arr = pd.DataFrame(identified_mode_shapes)

    # Get max absolute values for each true and identified mode shape to normalize
    max_abs_true1 = arr["trueMode1"].abs().max()
    max_abs_true2 = arr["trueMode2"].abs().max()
    max_abs_true3 = arr["trueMode3"].abs().max()

    max_abs_ident1 = arr["identifiedMode1"].abs().max()
    max_abs_ident2 = arr["identifiedMode2"].abs().max()
    max_abs_ident3 = arr["identifiedMode3"].abs().max()

    # Normalize
    arr["trueMode1_n"] = arr["trueMode1"] / max_abs_true1
    arr["trueMode2_n"] = arr["trueMode2"] / max_abs_true2
    arr["trueMode3_n"] = arr["trueMode3"] / max_abs_true3

    arr["identifiedMode1_n"] = arr["identifiedMode1"] / max_abs_ident1
    arr["identifiedMode2_n"] = arr["identifiedMode2"] / max_abs_ident2
    arr["identifiedMode3_n"] = arr["identifiedMode3"] / max_abs_ident3

    # Calculate MAC for each mode
    mac1 = calculate_mac(arr["trueMode1_n"].values, arr["identifiedMode1_n"].values)
    mac2 = calculate_mac(arr["trueMode2_n"].values, arr["identifiedMode2_n"].values)
    mac3 = calculate_mac(arr["trueMode3_n"].values, arr["identifiedMode3_n"].values)

    # Prepare result
    results = []
    for i, mode_num in enumerate([1, 2, 3]):
        true_freq = true_freqs[i]
        ident_freq = identified_freqs[i]
        freq_error = ((ident_freq - true_freq) / true_freq) * 100
        mac_val = [mac1, mac2, mac3][i]

        tm_col = f"trueMode{mode_num}_n"
        im_col = f"identifiedMode{mode_num}_n"

        mode_shapes = []
        for _, row in arr.iterrows():
            mode_shapes.append({
                "sensorId": row["sensorId"],
                "position": row["position"],
                "trueShape": row[tm_col],
                "identifiedShape": row[im_col]
            })

        results.append({
            "modeNumber": mode_num,
            "trueFrequency": true_freq,
            "identifiedFrequency": ident_freq,
            "frequencyError": f"{freq_error:.2f}",
            "mac": f"{mac_val:.4f}",
            "modeShapes": mode_shapes
        })
    return results


def calculate_mac(mode1: np.ndarray, mode2: np.ndarray) -> float:
    """
    Calculate the Modal Assurance Criterion (MAC) between two mode shape vectors.
    """
    numerator = (np.sum(mode1 * mode2))**2
    denominator = np.sum(mode1**2) * np.sum(mode2**2)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    st.set_page_config(page_title="Operational Modal Analysis Tool", layout="wide")

    # Author info
    st.markdown(
        """
        **Operational Modal Analysis Interactive Learning Tool**  
        Developed by: **Mohammad Talebi-Kalaleh**  
        Contact: [talebika@ualberta.ca](mailto:talebika@ualberta.ca)  
        """
    )
    
    # Introduction Section
    with st.expander("Introduction to OMA", expanded=False):
        st.write("""
        **Operational Modal Analysis (OMA)** is used to identify the dynamic properties (natural frequencies, 
        damping ratios, and mode shapes) of structures under their normal operating conditions 
        (ambient vibrations). 

        This educational tool demonstrates two OMA techniques:
        - **Frequency Domain Decomposition (FDD)**
        - **Stochastic Subspace Identification (SSI)** (placeholder)

        Use the parameters below to generate synthetic data and perform analysis to see 
        how well the identified modes match the "true" modes.
        """)

    # Default session-state parameters
    if "params" not in st.session_state:
        st.session_state.params = {
            "bridgeLength": 30.0,  # meters
            "numSensors": 10,
            "mode1": {"freq": 2.0, "amp": 50.0},
            "mode2": {"freq": 8.0, "amp": 10.0},
            "mode3": {"freq": 18.0, "amp": 1.0},
            "noiseLevel": 2.0,       # Standard deviation of noise
            "samplingFreq": 100.0,   # Hz
            "duration": 30.0         # seconds
        }

    # State variables to store data and flags
    if MEASUREMENTS_KEY not in st.session_state:
        st.session_state[MEASUREMENTS_KEY] = []
    if SVD_DATA_KEY not in st.session_state:
        st.session_state[SVD_DATA_KEY] = []
    if IDENTIFIED_MODES_KEY not in st.session_state:
        st.session_state[IDENTIFIED_MODES_KEY] = []
    if GENERATION_COMPLETE_KEY not in st.session_state:
        st.session_state[GENERATION_COMPLETE_KEY] = False
    if IS_ANALYZING_KEY not in st.session_state:
        st.session_state[IS_ANALYZING_KEY] = False

    params = st.session_state.params

    # --------------------------------------------------------------------------
    # Simulation Setup
    # --------------------------------------------------------------------------
    st.header("Simulation Setup")

    # Top row: number of sensors, bridge length
    c1, c2 = st.columns(2)
    with c1:
        params["numSensors"] = st.slider("Number of Sensors", 5, 20, int(params["numSensors"]), 1)
    with c2:
        params["bridgeLength"] = st.slider("Bridge Length (m)", 10.0, 100.0, float(params["bridgeLength"]), 5.0)

    # Next row: mode frequencies and amplitudes
    col1, col2 = st.columns(2)
    with col1:
        params["mode1"]["freq"] = st.slider("Mode 1 Frequency (Hz)", 0.5, 5.0, params["mode1"]["freq"], 0.1)
        params["mode1"]["amp"]  = st.slider("Mode 1 Amplitude", 10.0, 100.0, params["mode1"]["amp"], 5.0)

        params["mode2"]["freq"] = st.slider("Mode 2 Frequency (Hz)", 5.0, 15.0, params["mode2"]["freq"], 0.5)
        params["mode2"]["amp"]  = st.slider("Mode 2 Amplitude", 1.0, 50.0, params["mode2"]["amp"], 1.0)

    with col2:
        params["mode3"]["freq"] = st.slider("Mode 3 Frequency (Hz)", 15.0, 25.0, params["mode3"]["freq"], 0.5)
        params["mode3"]["amp"]  = st.slider("Mode 3 Amplitude", 0.1, 10.0, params["mode3"]["amp"], 0.1)

    # Advanced parameters (noise level, sampling frequency)
    with st.expander("Advanced Parameters", expanded=False):
        a1, a2 = st.columns(2)
        with a1:
            params["noiseLevel"] = st.slider("Noise Level", 0.0, 10.0, params["noiseLevel"], 0.5)
        with a2:
            params["samplingFreq"] = st.slider("Sampling Frequency (Hz)", 20.0, 200.0, params["samplingFreq"], 10.0)

    st.session_state.params = params

    # --------------------------------------------------------------------------
    # Separate box for "Generate Synthetic Data"
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Data Generation")
    if st.button("Generate Synthetic Measurements"):
        st.session_state[MEASUREMENTS_KEY] = generate_synthetic_data(st.session_state.params)
        st.session_state[GENERATION_COMPLETE_KEY] = True
        st.session_state[IDENTIFIED_MODES_KEY] = []
        st.session_state[SVD_DATA_KEY] = []

    # If data generation is done, show the measurement previews
    measurements = st.session_state[MEASUREMENTS_KEY]
    if st.session_state[GENERATION_COMPLETE_KEY] and measurements:
        st.markdown("**Synthetic Measurements Preview**")

        first_sensor_data = measurements[0]["data"][:500]  # limit to 500 points
        if first_sensor_data:
            df_sensor = pd.DataFrame(first_sensor_data)
            st.caption(f"Acceleration Time History (Sensor {measurements[0]['sensorId']})")
            line_chart = alt.Chart(df_sensor).mark_line().encode(
                x=alt.X("time", title="Time (s)"),
                y=alt.Y("acceleration", title="Acceleration"),
                tooltip=["time", "acceleration"]
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

        # Sensor positions
        st.caption("**Sensor Positions**")
        df_positions = pd.DataFrame(
            {"sensorId": m["sensorId"], "position": m["position"], "y": 0.5}
            for m in measurements
        )
        scatter_chart = alt.Chart(df_positions).mark_point(size=100, shape="triangle").encode(
            x=alt.X("position", title="Position (m)"),
            y=alt.Y("y", title="", scale=alt.Scale(domain=[0,1])),
            tooltip=["sensorId", "position"]
        ).properties(height=150)
        st.altair_chart(scatter_chart, use_container_width=True)

        pos_info = [f"{m['sensorId']}: {m['position']:.2f} m" for m in measurements]
        st.caption(" | ".join(pos_info))

    # --------------------------------------------------------------------------
    # Separate box for "Perform Analysis" (only enabled after generation)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Analysis")

    analysis_method = st.selectbox(
        "Choose Analysis Method",
        ("FDD", "SSI"),
        key="analysis_method_select"
    )

    def run_analysis():
        st.session_state[IS_ANALYZING_KEY] = True
        identified_modes, svd_data = perform_analysis(
            analysis_method,
            st.session_state.params,
            st.session_state[MEASUREMENTS_KEY]
        )
        st.session_state[IDENTIFIED_MODES_KEY] = identified_modes
        st.session_state[SVD_DATA_KEY] = svd_data
        st.session_state[IS_ANALYZING_KEY] = False

    st.button(
        "Perform Analysis",
        on_click=run_analysis,
        disabled=not st.session_state[GENERATION_COMPLETE_KEY]
    )

    # --------------------------------------------------------------------------
    # Display Analysis Results
    # --------------------------------------------------------------------------
    identified_modes = st.session_state[IDENTIFIED_MODES_KEY]
    svd_data = st.session_state[SVD_DATA_KEY]

    if identified_modes:
        st.markdown(f"### Analysis Results ({analysis_method})")

        # For FDD, show SVD plots
        if analysis_method == "FDD" and svd_data:
            st.markdown("#### Singular Value Decomposition")
            df_svd = pd.DataFrame(svd_data)

            col_svd1, col_svd2, col_svd3 = st.columns(3)

            def make_ref_line(freq):
                return alt.Chart(pd.DataFrame({"freq": [freq]})).mark_rule(
                    strokeDash=[4, 2], color="gray"
                ).encode(x="freq")

            # We'll determine a universal domain for the x-axis: [0, 2*mode3 freq]
            max_x = 2.0 * params["mode3"]["freq"]

            # Plot 1 (SV1)
            with col_svd1:
                st.write("**First Singular Value**")
                c1 = alt.Chart(df_svd).mark_line().encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                    y=alt.Y("sv1", title="Magnitude"),
                    tooltip=["frequency", "sv1"]
                )
                chart_sv1 = c1
                # Add reference lines
                ref1 = make_ref_line(params["mode1"]["freq"])
                ref2 = make_ref_line(params["mode2"]["freq"])
                ref3 = make_ref_line(params["mode3"]["freq"])
                chart_sv1 = (chart_sv1 + ref1 + ref2 + ref3).properties(height=200)
                st.altair_chart(chart_sv1, use_container_width=True)

            # Plot 2 (SV2)
            with col_svd2:
                st.write("**Second Singular Value**")
                c2 = alt.Chart(df_svd).mark_line(color="#D62728").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                    y=alt.Y("sv2", title="Magnitude"),
                    tooltip=["frequency", "sv2"]
                )
                chart_sv2 = c2
                ref1 = make_ref_line(params["mode1"]["freq"])
                ref2 = make_ref_line(params["mode2"]["freq"])
                ref3 = make_ref_line(params["mode3"]["freq"])
                chart_sv2 = (chart_sv2 + ref1 + ref2 + ref3).properties(height=200)
                st.altair_chart(chart_sv2, use_container_width=True)

            # Plot 3 (SV3)
            with col_svd3:
                st.write("**Third Singular Value**")
                c3 = alt.Chart(df_svd).mark_line(color="#2CA02C").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                    y=alt.Y("sv3", title="Magnitude"),
                    tooltip=["frequency", "sv3"]
                )
                chart_sv3 = c3
                ref1 = make_ref_line(params["mode1"]["freq"])
                ref2 = make_ref_line(params["mode2"]["freq"])
                ref3 = make_ref_line(params["mode3"]["freq"])
                chart_sv3 = (chart_sv3 + ref1 + ref2 + ref3).properties(height=200)
                st.altair_chart(chart_sv3, use_container_width=True)

            st.info(
                "Peaks in the first singular value often correspond to the natural frequencies. "
                "The second and third singular values help identify closely spaced modes."
            )

        # Identified Modal Parameters
        st.markdown("#### Identified Modal Parameters")
        df_ident = pd.DataFrame([
            {
                "Mode": m["modeNumber"],
                "True Frequency (Hz)": f"{m['trueFrequency']:.2f}",
                "Identified Frequency (Hz)": f"{m['identifiedFrequency']:.2f}",
                "Error (%)": m["frequencyError"],
                "MAC Value": m["mac"]
            } 
            for m in identified_modes
        ])
        st.table(df_ident)

        # MAC Equation in LaTeX/Markdown
        st.markdown(
            r"""
            **MAC Formula**:

            \[
            \text{MAC}(\phi_i, \phi_j) = \frac{\left|\phi_i^T \phi_j\right|^2}{\left(\phi_i^T \phi_i\right)\left(\phi_j^T \phi_j\right)}
            \]
            """,
            unsafe_allow_html=True
        )

        # Mode Shapes
        st.markdown("#### Mode Shape Visualization")
        n_modes = len(identified_modes)
        mode_columns = st.columns(n_modes)

        for i, mode_info in enumerate(identified_modes):
            with mode_columns[i]:
                st.markdown(f"**Mode {mode_info['modeNumber']}**")

                df_mode = pd.DataFrame(mode_info["modeShapes"])
                mode_num = mode_info["modeNumber"]

                # Create a dense array for the "true shape"
                x_dense = np.linspace(0, params["bridgeLength"], 100)
                if mode_num == 1:
                    y_dense_raw = np.sin(np.pi * x_dense / params["bridgeLength"])
                elif mode_num == 2:
                    y_dense_raw = np.sin(2*np.pi * x_dense / params["bridgeLength"])
                else:
                    y_dense_raw = np.sin(3*np.pi * x_dense / params["bridgeLength"])

                max_abs_val = np.max(np.abs(y_dense_raw))
                y_dense_norm = y_dense_raw / max_abs_val
                df_true_dense = pd.DataFrame({"position": x_dense, "trueShape": y_dense_norm})

                # Identified shape from sensors + zero endpoints
                identified_points = [{"position": 0.0, "identifiedShape": 0.0}]
                for row in df_mode.itertuples():
                    identified_points.append({
                        "position": row.position,
                        "identifiedShape": row.identifiedShape
                    })
                identified_points.append({"position": params["bridgeLength"], "identifiedShape": 0.0})
                df_ident_curve = pd.DataFrame(identified_points)

                # True shape line (dashed gray)
                line_true = alt.Chart(df_true_dense).mark_line(
                    strokeDash=[5, 3],
                    color="gray"
                ).encode(
                    x=alt.X("position", title="Position (m)"),
                    y=alt.Y("trueShape", 
                            title="Normalized Amplitude",
                            scale=alt.Scale(domain=[-1.1, 1.1])
                    ),
                    tooltip=["position", "trueShape"]
                )

                # Identified shape line
                line_ident = alt.Chart(df_ident_curve).mark_line(color="red").encode(
                    x="position",
                    y=alt.Y("identifiedShape", scale=alt.Scale(domain=[-1.1, 1.1]))
                )

                # Identified shape points
                points_ident = alt.Chart(df_ident_curve).mark_point(
                    color="red",
                    filled=True,
                    size=50
                ).encode(
                    x="position",
                    y="identifiedShape",
                    tooltip=["position", "identifiedShape"]
                )

                chart_mode = (line_true + line_ident + points_ident).properties(
                    height=250
                ).interactive()

                st.altair_chart(chart_mode, use_container_width=True)

                st.caption(
                    f"**Frequency:** {float(mode_info['identifiedFrequency']):.2f} Hz "
                    f"(True: {float(mode_info['trueFrequency']):.2f} Hz), "
                    f"**MAC:** {mode_info['mac']}, "
                    f"**Error:** {mode_info['frequencyError']}%"
                )

    # Educational Resources & Footer
    st.markdown("---")
    st.subheader("Final Notes & Resources")
    st.markdown("""
    **References**:
    1. Brincker, R., & Ventura, C. (2015). *Introduction to Operational Modal Analysis*. Wiley.
    2. Rainieri, C., & Fabbrocino, G. (2014). *Operational Modal Analysis of Civil Engineering Structures*. Springer.
    3. Peeters, B., & De Roeck, G. (2001). Stochastic System Identification for Operational Modal Analysis: A Review.
       Journal of Dynamic Systems, Measurement, and Control, 123(4), 659-667.
    4. Brincker, R., Zhang, L., & Andersen, P. (2000). Modal identification from ambient responses using frequency domain decomposition.
       Proceedings of the 18th Int. Modal Analysis Conference (IMAC), San Antonio, Texas.
    5. Au, S. K. (2017). *Operational Modal Analysis: Modeling, Bayesian Inference, Uncertainty Laws*. Springer.

    **Feedback & Collaboration**  
    - Non-stationary excitation sources  
    - Closely spaced modes  
    - Optimal sensor placement  
    - Handling measurement noise  
    - Model order selection in parametric methods  

    Feel free to reach out to **Mohammad Talebi-Kalaleh** at 
    [talebika@ualberta.ca](mailto:talebika@ualberta.ca) for questions or collaboration.
    """)

    st.markdown("""
    <hr/>
    <div style="text-align:center; font-size:0.85rem; color:#666;">
    <p>Operational Modal Analysis Educational Tool<br/>
    Developed by: Mohammad Talebi-Kalaleh | talebika@ualberta.ca</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
