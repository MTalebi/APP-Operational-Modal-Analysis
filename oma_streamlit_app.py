import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from typing import List, Dict, Tuple
from scipy.signal import welch
from numpy.linalg import svd, eig

# -----------------------------------------------------------------------------
# Author: Mohammad Talebi-Kalaleh
# Contact: talebika@ualberta.ca
# -----------------------------------------------------------------------------
# This code implements an Operational Modal Analysis (OMA) educational tool in 
# Python using Streamlit, illustrating Frequency Domain Decomposition (FDD) 
# and Stochastic Subspace Identification (SSI) with more realistic approaches.
# -----------------------------------------------------------------------------

# Session-state keys to store data across interactions
MEASUREMENTS_KEY = "measurements"
SVD_DATA_KEY = "svd_data"
IDENTIFIED_MODES_KEY = "identified_modes"
GENERATION_COMPLETE_KEY = "generation_complete"
IS_ANALYZING_KEY = "is_analyzing"


# -----------------------------------------------------------------------------
# Data Generation
# -----------------------------------------------------------------------------
def generate_synthetic_data(params: dict):
    """
    Generate synthetic acceleration measurements of a structure with 3 modes + noise.

    Args:
        params: Dictionary containing simulation parameters.

    Returns:
        measurements: A list of dictionaries, one per sensor:
           {
             "sensorId": str,
             "position": float,
             "data": [ { "time": float, "acceleration": float, ...}, ... ]
           }
    """
    # Unpack parameters
    bridge_length = params["bridgeLength"]
    num_sensors   = params["numSensors"]
    mode1         = params["mode1"]
    mode2         = params["mode2"]
    mode3         = params["mode3"]
    noise_level   = params["noiseLevel"]
    sampling_freq = params["samplingFreq"]
    duration      = params["duration"]

    dt = 1.0 / sampling_freq
    num_samples = int(np.floor(duration * sampling_freq))

    # Define sensor positions along the bridge (uniformly spaced)
    sensor_positions = np.linspace(
        bridge_length / (num_sensors + 1),
        bridge_length * (num_sensors / (num_sensors + 1)),
        num_sensors
    )

    # Create time vector
    time_vector = np.arange(num_samples) * dt

    # Create the three modal contributions in the time domain
    mode1_response = mode1["amp"] * np.sin(2.0 * np.pi * mode1["freq"] * time_vector)
    mode2_response = mode2["amp"] * np.sin(2.0 * np.pi * mode2["freq"] * time_vector)
    mode3_response = mode3["amp"] * np.sin(2.0 * np.pi * mode3["freq"] * time_vector)

    # Precompute theoretical mode shapes at each sensor position
    # (simple pinned-pinned assumption for demonstration)
    mode_shapes = []
    for x in sensor_positions:
        s1 = np.sin(np.pi * x / bridge_length)
        s2 = np.sin(2.0 * np.pi * x / bridge_length)
        s3 = np.sin(3.0 * np.pi * x / bridge_length)
        mode_shapes.append((s1, s2, s3))

    # Generate time-series for each sensor (modal superposition + noise)
    measurements = []
    for i, x in enumerate(sensor_positions):
        shape1, shape2, shape3 = mode_shapes[i]
        data_list = []

        for t_idx in range(num_samples):
            # Combine the three modal responses scaled by the sensor's mode-shape values
            m1 = shape1 * mode1_response[t_idx]
            m2 = shape2 * mode2_response[t_idx]
            m3 = shape3 * mode3_response[t_idx]

            # White noise
            noise = noise_level * (2.0 * np.random.rand() - 1.0)

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


# -----------------------------------------------------------------------------
# Frequency Domain Decomposition (FDD) - Realistic Implementation
# -----------------------------------------------------------------------------
def perform_fdd(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform Frequency Domain Decomposition on the measured signals.

    Steps:
      1. Arrange acceleration data into an (n_sensors x n_samples) matrix.
      2. Estimate Cross-Power Spectral Density (CPSD) matrix using Welch's method 
         (or direct FFT). For each frequency line, we get an n_sensors x n_sensors matrix.
      3. Perform SVD of the CPSD matrix at each frequency line -> singular values/vectors.
      4. Identify peaks in the first singular value near each "true" frequency to find 
         identified frequencies. 
      5. Extract the corresponding singular vectors at those frequencies as the 
         identified mode shapes.
      6. Return mode data (freq, shape, etc.) plus singular-value data for plotting.

    Args:
        params: Dictionary of OMA parameters.
        measurements: Synthetic sensor data.

    Returns:
        identified_modes: List of dictionaries describing each identified mode.
        svd_data: List of dictionaries containing frequencies and singular values 
                  (for plotting).
    """
    # --- 1. Build (n_sensors x n_samples) acceleration matrix ---
    n_sensors = len(measurements)
    # Use first sensor's time length as reference
    n_samples = len(measurements[0]["data"])

    sampling_freq = params["samplingFreq"]
    # Create an array for all sensors
    data_matrix = np.zeros((n_sensors, n_samples))
    for i, sens in enumerate(measurements):
        data_matrix[i, :] = [pt["acceleration"] for pt in sens["data"]]

    # --- 2. Estimate cross-power spectra: Gxx(f) ---
    # Use Welch's method to get a consistent frequency resolution
    # We'll use 'welch' from scipy.signal, computing cross-spectral densities.
    # A simpler approach is pairwise "csd" for each sensor pair, 
    # then assemble Gxx. For demonstration, we'll do it pairwise.
    # We store all frequencies + cross-spectra in a 3D array: Gxx[freq_index, i, j].
    # Also store the single-sided frequencies that Welch returns.

    # Frequency resolution:
    nperseg = 1024 if n_samples >= 1024 else n_samples
    # We will gather the cross-power spectral matrix for each frequency bin
    # from i=0..(n_sensors-1) to j=0..(n_sensors-1).
    # Note that for i=j we get the auto-power spectral densities (PSDs).

    # Because Welch's csd uses 'scipy.signal.csd(x, y)', we'll do pairwise computations.
    freqs = None
    # For storing cross-spectral matrices: shape => (n_freqs, n_sensors, n_sensors)
    Gxx_list = []

    # We must figure out the frequency axis from one pair, then fill in the rest
    for i in range(n_sensors):
        for j in range(i, n_sensors):
            f_ij, P_ij = welch(data_matrix[i, :], data_matrix[j, :],
                               fs=sampling_freq, nperseg=nperseg,
                               scaling='density', window='hann', detrend=False)
            if freqs is None:
                freqs = f_ij
                n_freqs = len(freqs)
                # Initialize the 3D array
                Gxx = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)

            # Fill in the symmetrical components
            # For real signals, Gxy = conj(Gyx), but welch might be real if i=j
            Gxx[:, i, j] = P_ij
            Gxx[:, j, i] = np.conjugate(P_ij)

    Gxx_list = Gxx

    # --- 3. Perform SVD of Gxx at each frequency line ---
    # We'll extract the first 3 singular values for plotting.
    svd_data = []
    for idx_f, f in enumerate(freqs):
        U, S, _ = svd(Gxx_list[idx_f, :, :], full_matrices=True)
        # We only store the first 3 singular values for plotting
        sv1 = np.abs(S[0]) if len(S) > 0 else 0.0
        sv2 = np.abs(S[1]) if len(S) > 1 else 0.0
        sv3 = np.abs(S[2]) if len(S) > 2 else 0.0

        svd_data.append({
            "frequency": f,
            "sv1": sv1,
            "sv2": sv2,
            "sv3": sv3
        })

    # --- 4. Identify peaks near each "true" frequency ---
    # For demonstration, we know we have 3 modes: mode1, mode2, mode3
    # We'll do a local search around each true freq ± a certain bandwidth.
    # Then the identified frequency is the local maximum of sv1 in that region.
    mode1_true = params["mode1"]["freq"]
    mode2_true = params["mode2"]["freq"]
    mode3_true = params["mode3"]["freq"]
    bandwidth  = 1.5  # search ±1.5 Hz around each true freq

    identified_freqs = []
    identified_shapes = []

    for true_f in [mode1_true, mode2_true, mode3_true]:
        f_min = max(true_f - bandwidth, 0.0)
        f_max = true_f + bandwidth

        # Extract the sub-array of frequencies within [f_min, f_max]
        # and find the index of the maximum sv1 in that region
        candidate_points = [(i, d) for i, d in enumerate(svd_data) 
                            if d["frequency"] >= f_min and d["frequency"] <= f_max]
        if not candidate_points:
            # If no data in that band, fallback to the nearest frequency
            # or just skip
            # We'll pick the closest freq
            best_idx = np.argmin([abs(d["frequency"] - true_f) for d in svd_data])
        else:
            # Among candidate_points, find the max of sv1
            best_idx = max(candidate_points, key=lambda x: x[1]["sv1"])[0]

        identified_freq = svd_data[best_idx]["frequency"]
        identified_freqs.append(identified_freq)

        # The singular vectors at that frequency define the mode shape
        # We'll do an SVD again (or we can store U from above if we prefer).
        U_best, S_best, V_best = svd(Gxx_list[best_idx, :, :], full_matrices=True)

        # The first singular vector (U[:, 0]) is the mode shape in output space
        mode_shape = U_best[:, 0].real  # take real part
        # We'll store this shape for subsequent MAC comparisons
        identified_shapes.append(mode_shape)

    # --- 5. Build identified mode dictionaries for each mode ---
    # We'll compare with the theoretical sensor-by-sensor shape
    # from the measurement definitions, in finalize_modes below.
    identified_modes = finalize_modes_fdd(measurements, identified_freqs, [mode1_true, mode2_true, mode3_true], identified_shapes)

    return identified_modes, svd_data


def finalize_modes_fdd(measurements, identified_freqs, true_freqs, identified_shapes):
    """
    Finalize the FDD results by comparing identified shapes vs. true shapes sensor-by-sensor,
    normalizing, and computing MAC.

    Args:
        measurements: The synthetic data (to get sensor positions & true shapes).
        identified_freqs: List of identified frequencies for modes 1, 2, 3.
        true_freqs: List of true frequencies [f1, f2, f3].
        identified_shapes: List of (n_sensors) arrays containing identified shapes from SVD 
                           for each identified frequency.

    Returns:
        A list of dictionaries describing each identified mode: 
        {
          "modeNumber": int,
          "trueFrequency": float,
          "identifiedFrequency": float,
          "frequencyError": str,
          "mac": str,
          "modeShapes": [ { "sensorId", "position", "trueShape", "identifiedShape" }, ... ]
        }
    """
    n_sensors = len(measurements)
    sensor_positions = [m["position"] for m in measurements]

    # Compute "true" shapes from geometry, 
    # same assumption: pinned-pinned, shape = sin(n * pi * x / L)
    L = measurements[0]["position"] * (n_sensors+1)/n_sensors  # or from params (bridgeLength)
    # Actually simpler: we can retrieve from the data directly (since we generated them),
    # but let's recalc for demonstration.
    
    def true_shape(mode_number, x):
        return np.sin(mode_number * np.pi * x / L)

    # Build table of shapes sensor-by-sensor
    # identified_shapes[i] is the shape for the i-th mode
    identified_modes = []
    for i, mode_num in enumerate([1, 2, 3]):
        freq_true = true_freqs[i]
        freq_ident = identified_freqs[i]
        # shape from SVD
        shape_ident = identified_shapes[i]

        # build sensor-based shape info
        sensor_data = []
        for s_idx, sens in enumerate(measurements):
            x_sens = sens["position"]
            # True shape (unnormalized)
            ts = true_shape(mode_num, x_sens)
            # Identified shape from the 1st singular vector => shape_ident[s_idx]
            sensor_data.append({
                "sensorId": sens["sensorId"],
                "position": x_sens,
                "trueShape": ts,
                "identifiedShape": shape_ident[s_idx]
            })

        # Normalize, compute MAC, frequency error
        # We'll do that with a helper:
        mode_dict = finalize_mode_entry(mode_num, freq_true, freq_ident, sensor_data)
        identified_modes.append(mode_dict)

    return identified_modes


# -----------------------------------------------------------------------------
# SSI Implementation (COV-SSI) - Realistic Illustration
# -----------------------------------------------------------------------------
def perform_ssi(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform Stochastic Subspace Identification (SSI) on the measured signals (COV-SSI variant).

    Steps (COV-SSI approach in a simplified educational manner):
      1. Collect multi-sensor data into array X (n_sensors x n_samples).
      2. Estimate correlation functions / Toeplitz matrix from the data (the 'R' matrix).
      3. Build block Hankel or Toeplitz matrix from R.
      4. Compute SVD to obtain extended observability matrix.
      5. Estimate system matrices (A, C) from it, then eigen-decomposition to get frequencies and mode shapes.
      6. Compare with true frequencies and shapes (like we do in FDD).

    Returns:
        identified_modes: List of identified mode dictionaries.
        svd_data: An empty or minimal list, since pure SSI doesn't produce "singular value vs frequency."
    """
    # 1. Build data matrix
    n_sensors = len(measurements)
    n_samples = len(measurements[0]["data"])
    X = np.zeros((n_sensors, n_samples))
    for i, sens in enumerate(measurements):
        X[i, :] = [pt["acceleration"] for pt in sens["data"]]

    # 2. Estimate correlation function. We'll do a naive approach:
    #    Let R[k] = E[X(t) X(t+k)^T],  k=0..max_lags
    #    We'll set max_lags = something like 2*n_sensors or a user-chosen factor.
    max_lags = 3 * n_sensors  # simplistic choice
    R = [np.zeros((n_sensors, n_sensors)) for _ in range(max_lags)]
    # For unbiased estimate, we sum over t for which t+k < n_samples
    for k in range(max_lags):
        valid_count = 0
        for t in range(n_samples - k):
            x_t   = X[:, t][:, None]      # shape (n_sensors,1)
            x_tk  = X[:, t+k][None, :]    # shape (1,n_sensors)
            R[k] += x_t @ x_tk
            valid_count += 1
        if valid_count > 0:
            R[k] /= valid_count

    # 3. Build Block Hankel/Toeplitz matrix from correlation slices
    #    Typical subspace approach: big Toeplitz of block-rows [R1 R2 ... R_i]
    #    We'll do a minimal version: pick i = j = 10 (model order ~ 6).
    i_block = 10
    # Construct an extended data matrix H0 from R[1..i_block].
    # shape => (n_sensors*i_block) x (n_sensors*i_block) if we do symmetrical, but let's keep it simple.
    H = []
    for row in range(i_block):
        # horizontally we stack R[row+1], R[row+2], ...
        row_blocks = [R[col+row+1] for col in range(i_block)]
        H.append(np.hstack(row_blocks))
    H0 = np.vstack(H)  # shape => (n_sensors*i_block, n_sensors*i_block)

    # 4. SVD of this Hankel-like matrix
    U, S, Vt = svd(H0, full_matrices=False)
    # Choose a model order. We expect 2*modes or 2*3=6. We'll just pick 6 for this example.
    model_order = 6
    U1 = U[:, :model_order]
    S1 = np.diag(S[:model_order])
    V1 = Vt[:model_order, :]

    # The extended observability matrix O_ext ~ U1 * sqrt(S1)
    # The extended controllability is ~ sqrt(S1)*V1
    O_ext = U1 @ (S1 ** 0.5)
    C = O_ext[0:n_sensors, :]  # The first block row is the output matrix
    # Next block row used to find A
    O_down = O_ext[n_sensors:, :]  # shift by one block row
    # Pseudoinverse
    O_up_pinv = np.linalg.pinv(O_ext[:-n_sensors, :])

    # The system matrix A is roughly O_down * O_up_pinv
    A_est = O_down @ O_up_pinv

    # 5. Eigen-decomposition to find frequencies + mode shapes
    #    The discrete-time poles are eigenvalues of A_est. Convert to continuous freq.
    eigvals, eigvecs = eig(A_est)
    dt = 1.0 / params["samplingFreq"]

    # Natural freq (Hz) from discrete pole lambda => freq = angle/(2*pi*dt)
    # Damping can also be extracted, but we'll skip for brevity. 
    # The mode shape is row-space of C * eigenvector
    # We'll identify 3 largest stable modes with positive freq below Nyquist, etc.
    # For demonstration, we just pick the 3 with largest imaginary parts or near the known frequencies.

    # Build a list of (freq_hz, shape_vector)
    identified_poles = []
    for idx, lam in enumerate(eigvals):
        mag = np.abs(lam)
        # If system stable => mag < 1. We'll keep it if 0<mag<1
        if mag < 1.0:
            # Discrete-time angle
            wn_dt = np.angle(lam)
            freq_hz = wn_dt / (2.0 * np.pi * dt)
            # Mode shape ~ C * eigenvector
            shape = C @ eigvecs[:, idx]
            identified_poles.append((freq_hz, shape))

    # Sort by ascending absolute frequency (just to pick the first 3 that match the "positive" freq)
    identified_poles.sort(key=lambda x: abs(x[0]))

    # If fewer than 3 stable poles, fill up with zeros. Otherwise pick 3 that
    # best match the known frequencies. We'll do a nearest approach around each true freq
    mode1_true = params["mode1"]["freq"]
    mode2_true = params["mode2"]["freq"]
    mode3_true = params["mode3"]["freq"]

    # We'll do a small function to pick best match near a true freq
    def pick_pole_near_freq(true_f, candidates, bandwidth=1.5):
        # find subset of candidates freq in [true_f - band, true_f + band]
        subset = [(fr, shp) for (fr, shp) in candidates if abs(fr - true_f) <= bandwidth]
        if not subset:
            # fallback to closest
            best = min(candidates, key=lambda x: abs(x[0] - true_f))
        else:
            # pick the largest magnitude shape or simply pick the freq closest to true_f
            best = min(subset, key=lambda x: abs(x[0] - true_f))
        return best

    # pick each mode from the candidate poles
    f1, shape1 = pick_pole_near_freq(mode1_true, identified_poles)
    f2, shape2 = pick_pole_near_freq(mode2_true, identified_poles)
    f3, shape3 = pick_pole_near_freq(mode3_true, identified_poles)

    # finalize
    identified_modes = finalize_modes_ssi(measurements,
                                          [f1, f2, f3],
                                          [mode1_true, mode2_true, mode3_true],
                                          [shape1, shape2, shape3])
    # For SSI we typically do not have a "singular values vs. frequency" plot 
    # like FDD, so we might return empty or some representation of singular values from H0.
    # We'll provide a minimal placeholder if we want to visualize something:
    svd_data = []
    for i, val in enumerate(np.diag(S1)):
        svd_data.append({"frequency": float(i), "sv1": float(val), "sv2": 0.0, "sv3": 0.0})

    return identified_modes, svd_data


def finalize_modes_ssi(measurements, identified_freqs, true_freqs, identified_shapes):
    """
    Finalize the SSI results, building identified mode dictionaries with MAC, frequency error, etc.

    Args:
        measurements: Synthetic data with sensor positions.
        identified_freqs: 3 identified frequencies from the subspace method.
        true_freqs: 3 known (true) frequencies from the simulation.
        identified_shapes: 3 arrays containing the identified mode shape in output space.

    Returns:
        A list of dictionaries describing each identified mode.
    """
    # We can do exactly the same approach as finalize_modes_fdd,
    # except that the identified shape dimension might differ if the model order 
    # is large. We'll assume dimension = n_sensors or a partial row of size n_sensors.

    def true_shape(m, x, L):
        return np.sin(m * np.pi * x / L)

    n_sensors = len(measurements)
    L = measurements[0]["position"] * (n_sensors+1)/n_sensors

    identified_modes = []
    for i, mode_num in enumerate([1, 2, 3]):
        freq_true  = true_freqs[i]
        freq_ident = identified_freqs[i]
        shape_ident = identified_shapes[i]

        # shape_ident might be longer than n_sensors if C is bigger, 
        # so we only take the first n_sensors
        shape_ident_sensors = shape_ident[:n_sensors]

        sensor_data = []
        for s_idx, sens in enumerate(measurements):
            x_sens = sens["position"]
            ts = true_shape(mode_num, x_sens, L)
            sensor_data.append({
                "sensorId": sens["sensorId"],
                "position": x_sens,
                "trueShape": ts,
                "identifiedShape": shape_ident_sensors[s_idx].real
            })

        # finalize
        mode_dict = finalize_mode_entry(mode_num, freq_true, freq_ident, sensor_data)
        identified_modes.append(mode_dict)

    return identified_modes


# -----------------------------------------------------------------------------
# Common Helper to finalize each Mode (normalization, MAC, frequency error)
# -----------------------------------------------------------------------------
def finalize_mode_entry(mode_number, freq_true, freq_ident, sensor_data):
    """
    Normalizes identified & true shapes, computes MAC, frequency error, 
    and builds a consistent dictionary for the mode results.
    """
    df = pd.DataFrame(sensor_data)

    # Normalization
    max_true = df["trueShape"].abs().max()
    max_ident = df["identifiedShape"].abs().max()
    if max_true < 1e-12:
        max_true = 1.0
    if max_ident < 1e-12:
        max_ident = 1.0

    df["trueShapeN"] = df["trueShape"] / max_true
    df["identifiedShapeN"] = df["identifiedShape"] / max_ident

    # MAC
    mac_val = calculate_mac(df["trueShapeN"].values, df["identifiedShapeN"].values)
    # Freq error
    freq_error_percent = ( (freq_ident - freq_true) / freq_true ) * 100.0 if freq_true != 0 else 0.0

    # Repackage sensor-based info
    mode_shapes_list = []
    for row in df.itertuples():
        mode_shapes_list.append({
            "sensorId": row.sensorId,
            "position": row.position,
            "trueShape": row.trueShapeN,
            "identifiedShape": row.identifiedShapeN
        })

    return {
        "modeNumber": mode_number,
        "trueFrequency": freq_true,
        "identifiedFrequency": freq_ident,
        "frequencyError": f"{freq_error_percent:.2f}",
        "mac": f"{mac_val:.4f}",
        "modeShapes": mode_shapes_list
    }


# -----------------------------------------------------------------------------
# MAC Calculation
# -----------------------------------------------------------------------------
def calculate_mac(mode1: np.ndarray, mode2: np.ndarray) -> float:
    """
    Compute the Modal Assurance Criterion (MAC) between two mode-shape vectors.

    MAC(phi1, phi2) = |phi1^T phi2|^2 / ( (phi1^T phi1)*(phi2^T phi2) )
    """
    numerator = (np.sum(mode1 * mode2)) ** 2
    denominator = np.sum(mode1**2) * np.sum(mode2**2)
    if denominator < 1e-16:
        return 0.0
    return float(numerator / denominator)


# -----------------------------------------------------------------------------
# Main Analysis Wrapper
# -----------------------------------------------------------------------------
def perform_analysis(analysis_method: str, params: dict, measurements: List[Dict]):
    """
    Wrapper to call the 'true' FDD or 'true' SSI method (not simplified).
    Returns the identified modes and SVD data (if applicable).
    """
    if analysis_method == "FDD":
        return perform_fdd(params, measurements)
    elif analysis_method == "SSI":
        return perform_ssi(params, measurements)
    else:
        return [], []


# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------
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
        **Operational Modal Analysis (OMA)** is used to identify the dynamic properties 
        (natural frequencies, damping ratios, and mode shapes) of structures 
        under their normal operating conditions (ambient vibrations). 

        This educational tool demonstrates two OMA techniques:
        - **Frequency Domain Decomposition (FDD)**:
          - Constructs cross-power spectral density (CPSD) matrices at each frequency 
            and uses singular value decomposition (SVD) to identify modes.
        - **Stochastic Subspace Identification (SSI)**:
          - Relies on time-domain correlations and state-space modeling to extract 
            system matrices and modal parameters.

        Use the parameters below to generate synthetic data and perform analysis 
        to see how well the identified modes match the "true" modes.
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

        first_sensor_data = measurements[0]["data"][:500]  # limit to 500 points for quick plotting
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

        # For FDD, show SVD plots of cross-power spectral decomposition
        if analysis_method == "FDD" and svd_data:
            st.markdown("#### Singular Value Decomposition (from CPSD)")
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
                # Add reference lines for the known true frequencies
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
                "FDD uses the cross-power spectral density (CPSD) matrix at each frequency. "
                "A peak in the first singular value often indicates a structural mode. "
                "The singular vectors provide the mode shapes."
            )

        # If SSI, we might have a minimal SVD plot or none. We'll just show the table normally.

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
            \text{MAC}(\phi_i, \phi_j) 
            = \frac{\left|\phi_i^T \phi_j\right|^2}
                   {\left(\phi_i^T \phi_i\right)\left(\phi_j^T \phi_j\right)}
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
                # We'll plot lines: the "true" shape and the "identified" shape. 
                # They are already normalized to [-1, 1], approximately.
                x_dense = np.linspace(0, params["bridgeLength"], 100)

                # Build an Altair chart for the identified shapes
                # We'll do the same approach: dashed line for "true shape" (interpolated),
                # a continuous line for the identified shape + sensor points. 
                # But we already have the discrete sensor data in df_mode. 
                # So let's just plot them as lines with a 0 at start and end.

                # "True shape" line: we can just connect the sensor positions 
                # or do a function-based approach. We'll do sensor-based here:
                # Insert a zero at start & end for a pinned shape visually:
                df_true_line = pd.concat([
                    pd.DataFrame({"position": [0], "trueShapeN": [0]}),
                    df_mode[["position", "trueShape"]].rename(columns={"trueShape": "trueShapeN"}),
                    pd.DataFrame({"position": [params["bridgeLength"]], "trueShapeN": [0]})
                ], ignore_index=True)

                df_ident_line = pd.concat([
                    pd.DataFrame({"position": [0], "identifiedShapeN": [0]}),
                    df_mode[["position", "identifiedShape"]].rename(columns={"identifiedShape": "identifiedShapeN"}),
                    pd.DataFrame({"position": [params["bridgeLength"]], "identifiedShapeN": [0]})
                ], ignore_index=True)

                line_true = alt.Chart(df_true_line).mark_line(
                    strokeDash=[5, 3],
                    color="gray"
                ).encode(
                    x=alt.X("position", title="Position (m)"),
                    y=alt.Y("trueShapeN", 
                            title="Normalized Amplitude",
                            scale=alt.Scale(domain=[-1.1, 1.1])
                    )
                )

                line_ident = alt.Chart(df_ident_line).mark_line(color="red").encode(
                    x="position",
                    y=alt.Y("identifiedShapeN", scale=alt.Scale(domain=[-1.1, 1.1]))
                )
                points_ident = alt.Chart(df_mode).mark_point(
                    color="red",
                    filled=True,
                    size=50
                ).encode(
                    x="position",
                    y=alt.Y("identifiedShape", scale=alt.Scale(domain=[-1.1, 1.1])),
                    tooltip=["sensorId", "identifiedShape"]
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
    3. Peeters, B., & De Roeck, G. (2001). *Stochastic System Identification for Operational Modal Analysis: A Review*.
       Journal of Dynamic Systems, Measurement, and Control, 123(4), 659-667.
    4. Brincker, R., Zhang, L., & Andersen, P. (2000). *Modal identification from ambient responses using frequency domain decomposition*.
       In Proceedings of the 18th Int. Modal Analysis Conference (IMAC), San Antonio, Texas.
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
    Developed by: Mohammad Talebi-Kalaleh | <a href="mailto:talebika@ualberta.ca">talebika@ualberta.ca</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
