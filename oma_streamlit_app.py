import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

#####################################################################
# -------------- CESSIPy SSI REFERENCE CODE (ABRIDGED) --------------
#####################################################################

# Below is an abridged subset of the reference code you provided, adapted
# to our needs. We have:
#   - MRPy class for time-data
#   - Key Covariance-Driven SSI functions: Toeplitz(...) + SSI_COV(...) or Fast_SSI(..., based='COV')
#   - A minimal approach to pick the top modes near known freq.

class auxclass(np.ndarray):
    """Simple class to store additional attributes on top of an ndarray."""
    def __new__(cls, np_array):
        return np.asarray(np_array).view(cls)

class MRPy(np.ndarray):
    """
    Multivariate Random Process with uniform time step, from CESSIPy references.
    This is a minimal version just enough for the covariance-driven SSI.
    """
    def __new__(cls, data, fs=None):
        obj = np.asarray(data).view(cls)
        obj.fs = fs
        obj.NX = obj.shape[0]  # number of channels (rows)
        obj.N  = obj.shape[1]  # number of samples (columns)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.fs = getattr(obj, 'fs', None)
        self.NX = getattr(obj, 'NX', None)
        self.N  = getattr(obj, 'N', None)

def rearrange_data(yk, ref):
    """
    Reorder channels so that the 'ref' sensor indices come first, 
    matching the logic from CESSIPy. For advanced usage with multiple references.
    """
    r = len(ref)
    l = yk.shape[0]
    # new MRPy with same # of samples, reorder rows
    new_data = np.empty((l, yk.shape[1]))
    new_data[:r,:] = yk[ref,:]
    keep_others = [i for i in range(l) if i not in ref]
    new_data[r:,:] = yk[keep_others,:]

    yk2 = MRPy(new_data, fs=yk.fs)
    yk2.r = r
    yk2.l = l
    return yk2

def Toeplitz(yk, i):
    """
    Build the block Toeplitz matrix from the data yk, with i time-lags.
    """
    N = yk.N - 2*i + 1
    # r, l set from rearrange_data or manually
    r = getattr(yk, 'r', 1)  # default 1 reference if not defined
    l = getattr(yk, 'l', yk.NX)

    # Partition data into Ypref (reference block) and Yf (full block)
    Ypref = np.zeros((r*i, N))
    Yf    = np.zeros((l*i, N))
    for k in range(i):
        Ypref[k*r:(k+1)*r,:] = yk[:r,    k:k+N ]
        Yf   [k*l:(k+1)*l,:] = yk[   :, k+i:k+i+N]

    # Scale for unbiased
    Ypref /= np.sqrt(N)
    Yf    /= np.sqrt(N)

    T = auxclass(Yf @ Ypref.T)
    # store meta info
    T.fs = yk.fs
    T.r  = r
    T.l  = l
    T.i  = i
    return T

def SSI_COV(T, no):
    """
    Covariance-driven subspace ID on matrix T (Toeplitz).
    no = state-space model order.
    Returns freq, damping, and mode-shapes.
    """
    # 1. SVD of T
    U, S, Vt = T.SVD
    # Truncate to 'no'
    U1 = U[:,:no]
    S1 = np.diag(S[:no])
    # Extended observability
    Oi = U1 @ np.sqrt(S1)
    l = T.l
    i = T.i
    # C = first block row
    C  = Oi[:l, :]
    # separate for next block row
    Oi_down = Oi[l:, :]
    # Pseudoinverse of the upper portion
    Oi_up_pinv = np.linalg.pinv(Oi[:l*(i-1), :])
    # System matrix A
    A_est = Oi_down @ Oi_up_pinv

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eig(A_est)
    # dt from sampling freq
    dt = 1.0 / T.fs
    # discrete poles => lam
    lam = np.log(eigvals) * T.fs  # for natural freq
    # natural freq in Hz
    fn = np.abs(lam) / (2.0 * np.pi)
    zt = -np.real(lam) / np.abs(lam)  # damping ratio
    # mode shapes => C @ eigvec
    shapes = C @ eigvecs
    return fn, zt, shapes

def SSI_COV_iterator(yk, i, nmin, nmax, incr=2):
    """
    For demonstration, we do an iterator for multiple model orders.
    Return all frequencies, damping, shapes for each order.
    """
    T  = Toeplitz(yk, i)
    # SVD once for T
    T.SVD = np.linalg.svd(T, full_matrices=False)
    orders = np.arange(nmin, nmax+1, incr)
    # We'll store them
    freq_list = []
    damp_list = []
    shape_list= []
    for no in orders:
        fn, zt, V = SSI_COV(T, no)
        freq_list.append(fn)
        damp_list.append(zt)
        shape_list.append(V)
    return orders, freq_list, damp_list, shape_list

#####################################################################
# -------------- END OF CESSIPy SSI REFERENCE CODE ------------------
#####################################################################


################################################################################
# -------------- YOUR FDDAuto CODE (FDDAuto-BASED FDD) from previous -----------
################################################################################

import gzip as gz
import pickle as pk
from scipy.signal import csd as cpsd
from scipy.fftpack import fft, fftfreq
from scipy.linalg import svd

def cpsd_fft_based(x, y, Fs):
    # Simple placeholder if needed. Usually we won't use it.
    n = len(x)
    X = fft(x, n=n)
    Y = fft(y, n=n)
    Gxy = (X * np.conjugate(Y)) / n
    freq = fftfreq(n, d=1.0 / Fs)
    half = n // 2 + 1
    return freq[:half], Gxy[:half]

def FDDAuto(Acc, Fs, Num_Modes_Considered, EstimeNaturalFreqs, set_singular_ylim=None, csd_method='cpsd'):
    """
    Frequency Domain Decomposition (FDD) algorithm for modal analysis (original snippet).
    ...
    (Same as you've provided)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import csd as cpsd

    PSD = np.zeros((Acc.shape[0] // 2 + 1, Acc.shape[1], Acc.shape[1]), dtype=complex)
    F = np.zeros((Acc.shape[0] // 2 + 1, Acc.shape[1], Acc.shape[1]))

    for I in range(Acc.shape[1]):
        for J in range(Acc.shape[1]):
            if csd_method == 'cpsd':
                f_ij, PSD_ij = cpsd(
                    Acc[:, I], Acc[:, J],
                    nperseg=Acc.shape[0] // 3,
                    noverlap=None,
                    nfft=Acc.shape[0],
                    fs=Fs
                )
            else:
                f_ij, PSD_ij = cpsd_fft_based(Acc[:, I], Acc[:, J], Fs)
            F[:, I, J] = f_ij
            PSD[:, I, J] = PSD_ij

    PSD_list = []
    F_list   = []
    for K in range(PSD.shape[0]):
        PSD_list.append(PSD[K, :, :])
        F_list.append(F[K, :, :])

    # Then we do the Identifier step (SVD, picking peaks, etc.)
    Frq, Phi_Real, Fp, s1 = Identifier(PSD_list, F_list, Num_Modes_Considered, EstimeNaturalFreqs, set_singular_ylim)
    return Frq, Phi_Real

def Identifier(PSD, F, Num_Modes_Considered, EstimeNaturalFreqs, set_singular_ylim):
    """
    ...
    The same code you provided for peak picking, using SVD at each freq line.
    ...
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import svd

    s1 = np.zeros(len(PSD))
    s2 = np.zeros(len(PSD))
    ms = np.zeros((PSD[0].shape[0], len(PSD)))
    for K in range(len(PSD)):
        PSD_2DMatrix = PSD[K]
        u, s, _ = svd(PSD_2DMatrix)
        s1[K] = s[0]
        if len(s) > 1:
            s2[K] = s[1]
        ms[:, K] = u[:, 0]

    Freq = np.stack(F)[:,1,1].flatten()
    s1_db = 10 * np.log10(s1)
    Fp = []
    NumPeaks = Num_Modes_Considered

    bandwidth = 0.3
    # pick peaks near each EstimeNaturalFreqs[k]
    for k in range(NumPeaks):
        f_est = EstimeNaturalFreqs[k]
        lowb = f_est - bandwidth
        highb = f_est + bandwidth
        if lowb < 0: lowb = 0
        subset_idx = np.where((Freq >= lowb) & (Freq <= highb))[0]
        if len(subset_idx) == 0:
            # fallback
            best_idx = np.argmin(np.abs(Freq - f_est))
        else:
            best_idx = subset_idx[np.argmax(s1[subset_idx])]
        Fp.append([best_idx, Freq[best_idx]])

    Fp = np.array(Fp)
    # sort
    sr = np.argsort(Fp[:,1])
    Fp = Fp[sr,:]

    # produce figure
    fig = plt.figure(figsize=(8,5), dpi=100)
    ax = fig.subplots()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('1st Singular Value (dB)')
    ax.plot(Freq, s1_db, linewidth=2, color='k', linestyle='-')
    ax.grid(True)
    ax.set_xlim([0, Fp[-1,1] * 3.0])
    for i in range(Fp.shape[0]):
        idx_peak = int(Fp[i,0])
        freq_peak = Fp[i,1]
        ax.scatter(freq_peak, s1_db[idx_peak], marker='o', edgecolors='r', facecolors='r')
        ax.axvline(x=freq_peak, color='red', linestyle='--')
        ax.annotate(f'f_{i+1}={round(freq_peak,2)}',
                    xy=(freq_peak, s1_db[idx_peak]),
                    xytext=(-25, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->"),
                    fontsize=13)
    if set_singular_ylim is not None:
        ax.set_ylim(set_singular_ylim)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    ax.tick_params(labelsize=12)
    fig.tight_layout()

    # identified freq
    NaturalIDFrq = Fp[:,1]

    # compute mode shapes at each peak
    from scipy.linalg import svd
    Phi_Complex = np.zeros((PSD[0].shape[0], NumPeaks), dtype=complex)
    Phi_Real    = np.zeros((PSD[0].shape[0], NumPeaks))
    for j in range(NumPeaks):
        peak_idx = int(Fp[j,0])
        PSD_2DMatrix = PSD[peak_idx]
        ug = np.linalg.svd(PSD_2DMatrix)
        u_peak, s_peak, vh_peak = ug
        phi_cpx = u_peak[:, 0]
        # align phase
        Y = np.imag(phi_cpx)
        X = np.column_stack((np.real(phi_cpx), np.ones(len(phi_cpx))))
        A = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
        theta = np.arctan(A[0])
        phi_rot = phi_cpx * np.exp(-1j*theta)
        phi_rot_real = np.real(phi_rot)
        phi_rot_real /= np.max(np.abs(phi_rot_real))
        Phi_Complex[:, j] = phi_cpx
        Phi_Real   [:, j] = phi_rot_real

    return NaturalIDFrq, Phi_Real, Fp, s1


def identify_mode_shapes_frqs(D, num_modes_considered, sampling_rate, estimated_natural_freqs, method='fdd', set_singular_ylim='yes'):
    """
    Minimal wrapper for FDDAuto in a separate function if needed.
    """
    identified_natural_frqs = []
    identified_modes_shapes = None
    if method == 'fdd':
        identified_natural_frqs, identified_modes_shapes = FDDAuto(D.T, sampling_rate, num_modes_considered, estimated_natural_freqs, set_singular_ylim)
    return identified_natural_frqs, identified_modes_shapes


################################################################################
# -------------- GENERATE SYNTHETIC DATA / COMMON FUNCTIONS --------------------
################################################################################

def generate_synthetic_data(params: dict):
    """
    Generate synthetic acceleration data for 3 modes + noise, 
    same as before.
    """
    import numpy as np
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

    sensor_positions = np.linspace(
        bridge_length / (num_sensors + 1),
        bridge_length * (num_sensors / (num_sensors + 1)),
        num_sensors
    )
    time_vector = np.arange(num_samples) * dt

    mode1_response = mode1["amp"] * np.sin(2.0*np.pi*mode1["freq"]*time_vector)
    mode2_response = mode2["amp"] * np.sin(2.0*np.pi*mode2["freq"]*time_vector)
    mode3_response = mode3["amp"] * np.sin(2.0*np.pi*mode3["freq"]*time_vector)

    mode_shapes = []
    for x in sensor_positions:
        s1 = np.sin(np.pi*x/bridge_length)
        s2 = np.sin(2.0*np.pi*x/bridge_length)
        s3 = np.sin(3.0*np.pi*x/bridge_length)
        mode_shapes.append((s1,s2,s3))

    measurements = []
    for i, x in enumerate(sensor_positions):
        shape1, shape2, shape3 = mode_shapes[i]
        data_list = []
        for t_idx in range(num_samples):
            m1 = shape1*mode1_response[t_idx]
            m2 = shape2*mode2_response[t_idx]
            m3 = shape3*mode3_response[t_idx]
            noise = noise_level*(2*np.random.rand()-1)
            total_acc = m1+m2+m3+noise
            data_list.append({
                "time": time_vector[t_idx],
                "acceleration": total_acc,
                "mode1": m1,
                "mode2": m2,
                "mode3": m3,
                "noise": noise
            })
        measurements.append({
            "sensorId": f"S{i+1}",
            "position": x,
            "data": data_list
        })
    return measurements


def calculate_mac(mode1: np.ndarray, mode2: np.ndarray) -> float:
    numerator = (np.sum(mode1 * mode2))**2
    denominator = np.sum(mode1**2)*np.sum(mode2**2)
    if denominator < 1e-16:
        return 0.0
    return float(numerator/denominator)

def finalize_mode_entry(mode_number, freq_true, freq_ident, sensor_data):
    import pandas as pd
    df = pd.DataFrame(sensor_data)
    max_true = df["trueShape"].abs().max()
    max_ident= df["identifiedShape"].abs().max()
    if max_true<1e-12: max_true=1.0
    if max_ident<1e-12: max_ident=1.0
    df["trueShapeN"] = df["trueShape"]/max_true
    df["identifiedShapeN"] = df["identifiedShape"]/max_ident

    mac_val = calculate_mac(df["trueShapeN"].values, df["identifiedShapeN"].values)
    freq_error_percent = ((freq_ident-freq_true)/freq_true*100.0) if freq_true!=0 else 0.0

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


################################################################################
# -------------- PERFORM FDD / PERFORM SSI (REAL) ------------------------------
################################################################################

def perform_fdd(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], None]:
    """
    FDD using FDDAuto snippet. Plots singular values with Matplotlib.
    Returns identified_modes, and svd_data is None or empty because 
    FDDAuto does the plotting internally.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n_sensors = len(measurements)
    n_samples = len(measurements[0]["data"])
    data_matrix = np.zeros((n_samples, n_sensors))
    for i, sens in enumerate(measurements):
        data_matrix[:, i] = [pt["acceleration"] for pt in sens["data"]]

    sampling_freq = params["samplingFreq"]
    # "true" frequencies as estimates
    mode1_f = params["mode1"]["freq"]
    mode2_f = params["mode2"]["freq"]
    mode3_f = params["mode3"]["freq"]
    estimated_frqs = [mode1_f, mode2_f, mode3_f]

    # call FDDAuto
    with plt.ioff():
        Frq, Phi_Real = FDDAuto(data_matrix, sampling_freq, 3, estimated_frqs)
        fig = plt.gcf()
    # display
    st.pyplot(fig)

    # finalize
    identified_modes = []
    sensor_positions = [m["position"] for m in measurements]
    true_freqs = [mode1_f, mode2_f, mode3_f]
    for i, mode_num in enumerate([1,2,3]):
        freq_true = true_freqs[i]
        freq_ident= Frq[i]
        shape_ident = Phi_Real[:,i]

        sensor_data = []
        for s_idx, sens in enumerate(measurements):
            x_sens = sens["position"]
            L = params["bridgeLength"]
            true_val = np.sin(mode_num*np.pi*x_sens/L)
            sensor_data.append({
                "sensorId": sens["sensorId"],
                "position": x_sens,
                "trueShape": true_val,
                "identifiedShape": shape_ident[s_idx]
            })
        mode_dict = finalize_mode_entry(mode_num, freq_true, freq_ident, sensor_data)
        identified_modes.append(mode_dict)

    return identified_modes, []


def perform_ssi(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], None]:
    """
    Real SSI approach using the Covariance-driven subspace ID from the 
    reference code. We'll pick a model order (like 6), then pick the 
    top 3 modes near the known frequencies.

    Steps:
       1. Build data in MRPy format: shape=(n_channels, n_samples).
       2. Possibly reorder reference sensors if needed, or just set ref=[0].
       3. Build Toeplitz with i=some int (like 20).
       4. SVD -> A_est -> freq and shapes.
       5. Pick out 3 modes near the known frequencies.
    """
    import numpy as np

    n_sensors = len(measurements)
    n_samples = len(measurements[0]["data"])

    # build data matrix shape => (n_sensors, n_samples)
    data_mat = np.zeros((n_sensors, n_samples))
    for i, sens in enumerate(measurements):
        data_mat[i,:] = [pt["acceleration"] for pt in sens["data"]]

    sampling_freq = params["samplingFreq"]
    # Create MRPy object
    yk = MRPy(data_mat, fs=sampling_freq)
    # For a single reference approach, we might do rearrange_data(yk, [0]) or so
    # but let's skip for now. If needed, set yk.r=1, yk.l=n_sensors
    yk.r = 1
    yk.l = n_sensors

    # Toeplitz parameters
    i_lags = 20   # user-chosen, tweak as needed
    # Single SVD
    T = Toeplitz(yk, i_lags)
    T.SVD = np.linalg.svd(T, full_matrices=False)

    # Pick a model order. We'll do no=6 for demonstration.
    no = 6
    fn, zt, shapes = SSI_COV(T, no)

    # Now we have up to 'no' modes in fn, zt, shapes (size: n_sensors x no)
    # We want the first 3 modes near the known frequencies:
    # We'll do a local search near each freq ± 1.0 Hz
    # Or simpler: pick the best match for each known freq by minimal abs difference.

    mode1_f = params["mode1"]["freq"]
    mode2_f = params["mode2"]["freq"]
    mode3_f = params["mode3"]["freq"]
    known_freqs = [mode1_f, mode2_f, mode3_f]

    identified_modes = []
    for i, mode_num in enumerate([1,2,3]):
        freq_true = known_freqs[i]
        # find the index in 'fn' that best matches freq_true
        # or if freq_true is out of band, fallback
        idx_best = np.argmin(np.abs(fn - freq_true))
        freq_ident = fn[idx_best]
        shape_ident = shapes[:, idx_best]

        # build sensor-based data
        sensor_data = []
        for s_idx, sens in enumerate(measurements):
            x_sens = sens["position"]
            L = params["bridgeLength"]
            true_val = np.sin(mode_num*np.pi*x_sens/L)
            sensor_data.append({
                "sensorId": sens["sensorId"],
                "position": x_sens,
                "trueShape": true_val,
                "identifiedShape": shape_ident[s_idx].real  # real part
            })

        mode_dict = finalize_mode_entry(mode_num, freq_true, freq_ident, sensor_data)
        identified_modes.append(mode_dict)

    # For a more advanced approach, you might do a "stabilization diagram" and pick stable poles, etc.
    # For now, we do a naive picking. We won't produce an svd_data because we don't have
    # a freq-domain SVD to plot.

    return identified_modes, []


def perform_analysis(analysis_method: str, params: dict, measurements: List[Dict]):
    """
    Final dispatcher: calls FDD or the new advanced SSI code.
    """
    if analysis_method=="FDD":
        return perform_fdd(params, measurements)
    elif analysis_method=="SSI":
        return perform_ssi(params, measurements)
    else:
        return [], []

################################################################################
# -------------- STREAMLIT APP LAYOUT -----------------------------------------
################################################################################

def main():
    st.set_page_config(page_title="Operational Modal Analysis Tool", layout="wide")

    # Author info
    st.markdown("""
    **Operational Modal Analysis Interactive Learning Tool**  
    Developed by: **Mohammad Talebi-Kalaleh**  
    Contact: [talebika@ualberta.ca](mailto:talebika@ualberta.ca)  
    """)

    with st.expander("Introduction to OMA", expanded=False):
        st.write("""
        **Operational Modal Analysis (OMA)** aims to identify the dynamic properties 
        (natural frequencies, damping ratios, and mode shapes) of structures 
        using only the measured output data under ambient excitations.

        This educational tool demonstrates two major OMA techniques:
        - **Frequency Domain Decomposition (FDD)**: We use your FDDAuto routine with cross-power spectral density + SVD
        - **Stochastic Subspace Identification (SSI)**: Covariance-driven subspace approach from the reference code (CESSIPy).

        Adjust the parameters below, generate synthetic data, and compare the identified 
        modes with the known “true” modes.
        """)

    # Default session-state
    if "params" not in st.session_state:
        st.session_state.params = {
            "bridgeLength": 30.0,
            "numSensors": 10,
            "mode1": {"freq": 2.0, "amp": 50.0},
            "mode2": {"freq": 8.0, "amp": 10.0},
            "mode3": {"freq": 18.0, "amp": 1.0},
            "noiseLevel": 2.0,
            "samplingFreq": 100.0,
            "duration": 30.0
        }

    if "measurements" not in st.session_state:
        st.session_state["measurements"] = []
    if "identified_modes" not in st.session_state:
        st.session_state["identified_modes"] = []
    if "svd_data" not in st.session_state:
        st.session_state["svd_data"] = []
    if "generation_complete" not in st.session_state:
        st.session_state["generation_complete"] = False
    if "is_analyzing" not in st.session_state:
        st.session_state["is_analyzing"] = False

    params = st.session_state.params

    st.header("Simulation Setup")
    c1, c2 = st.columns(2)
    with c1:
        params["numSensors"] = st.slider("Number of Sensors", 5, 20, int(params["numSensors"]), 1)
    with c2:
        params["bridgeLength"] = st.slider("Bridge Length (m)", 10.0, 100.0, float(params["bridgeLength"]), 5.0)

    col1, col2 = st.columns(2)
    with col1:
        params["mode1"]["freq"] = st.slider("Mode 1 Frequency (Hz)", 0.5, 5.0, params["mode1"]["freq"], 0.1)
        params["mode1"]["amp"]  = st.slider("Mode 1 Amplitude", 10.0, 100.0, params["mode1"]["amp"], 5.0)

        params["mode2"]["freq"] = st.slider("Mode 2 Frequency (Hz)", 5.0, 15.0, params["mode2"]["freq"], 0.5)
        params["mode2"]["amp"]  = st.slider("Mode 2 Amplitude", 1.0, 50.0, params["mode2"]["amp"], 1.0)

    with col2:
        params["mode3"]["freq"] = st.slider("Mode 3 Frequency (Hz)", 15.0, 25.0, params["mode3"]["freq"], 0.5)
        params["mode3"]["amp"]  = st.slider("Mode 3 Amplitude", 0.1, 10.0, params["mode3"]["amp"], 0.1)

    with st.expander("Advanced Parameters", expanded=False):
        a1, a2 = st.columns(2)
        with a1:
            params["noiseLevel"] = st.slider("Noise Level", 0.0, 10.0, params["noiseLevel"], 0.5)
        with a2:
            params["samplingFreq"] = st.slider("Sampling Frequency (Hz)", 20.0, 200.0, params["samplingFreq"], 10.0)

    st.session_state.params = params

    st.markdown("---")
    st.subheader("Data Generation")

    if st.button("Generate Synthetic Measurements"):
        st.session_state["measurements"] = generate_synthetic_data(st.session_state.params)
        st.session_state["generation_complete"] = True
        st.session_state["identified_modes"] = []
        st.session_state["svd_data"] = []

    measurements = st.session_state["measurements"]
    if st.session_state["generation_complete"] and measurements:
        st.markdown("**Synthetic Measurements Preview**")
        # show time-history from sensor 1
        first_data = measurements[0]["data"][:500]
        if first_data:
            df = pd.DataFrame(first_data)
            st.caption(f"Acceleration Time History (Sensor {measurements[0]['sensorId']})")
            line_chart = alt.Chart(df).mark_line().encode(
                x=alt.X("time", title="Time (s)"),
                y=alt.Y("acceleration", title="Acceleration"),
                tooltip=["time", "acceleration"]
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

        # sensor positions
        st.caption("**Sensor Positions**")
        df_pos = pd.DataFrame({
            "sensorId": [m["sensorId"] for m in measurements],
            "position": [m["position"] for m in measurements],
            "y": [0.5]*len(measurements)
        })
        scatter_chart = alt.Chart(df_pos).mark_point(size=100, shape="triangle").encode(
            x=alt.X("position", title="Position (m)"),
            y=alt.Y("y", title="", scale=alt.Scale(domain=[0,1])),
            tooltip=["sensorId", "position"]
        ).properties(height=150)
        st.altair_chart(scatter_chart, use_container_width=True)

        txt = " | ".join([f"{m['sensorId']}: {m['position']:.2f} m" for m in measurements])
        st.caption(txt)


    st.markdown("---")
    st.subheader("Analysis")

    analysis_method = st.selectbox(
        "Choose Analysis Method",
        ("FDD", "SSI"),
        key="analysis_method_select"
    )

    def run_analysis():
        st.session_state["is_analyzing"] = True
        identified_modes, svd_data = perform_analysis(
            analysis_method,
            st.session_state.params,
            st.session_state["measurements"]
        )
        st.session_state["identified_modes"] = identified_modes
        st.session_state["svd_data"] = svd_data
        st.session_state["is_analyzing"] = False

    st.button("Perform Analysis", on_click=run_analysis, disabled=not st.session_state["generation_complete"])

    identified_modes = st.session_state["identified_modes"]
    svd_data         = st.session_state["svd_data"]

    if identified_modes:
        st.markdown(f"### Analysis Results ({analysis_method})")

        # If FDD, we have already displayed the singular value plot (Matplotlib).
        # If SSI, no direct singular-value plot but we do show the table below.

        st.markdown("#### Identified Modal Parameters")
        df_modes = pd.DataFrame([
            {
                "Mode": m["modeNumber"],
                "True Frequency (Hz)": f"{m['trueFrequency']:.2f}",
                "Identified Frequency (Hz)": f"{m['identifiedFrequency']:.2f}",
                "Error (%)": m["frequencyError"],
                "MAC Value": m["mac"]
            }
            for m in identified_modes
        ])
        st.table(df_modes)

        st.latex(
            r"""
            \text{MAC} =
            \frac{\left| \phi_i^T \phi_j \right|^2}
                 {\left( \phi_i^T \phi_i \right) \left( \phi_j^T \phi_j \right)}
            """
        )

        st.markdown("#### Mode Shape Visualization")
        n_modes = len(identified_modes)
        mode_cols = st.columns(n_modes)

        for i, mode_info in enumerate(identified_modes):
            with mode_cols[i]:
                st.markdown(f"**Mode {mode_info['modeNumber']}**")
                df_mode = pd.DataFrame(mode_info["modeShapes"])
                # Build pinned shape line
                # Insert zero endpoints
                df_true_line = pd.concat([
                    pd.DataFrame({"position":[0], "trueShapeN":[0]}),
                    df_mode[["position","trueShape"]].rename(columns={"trueShape":"trueShapeN"}),
                    pd.DataFrame({"position":[params["bridgeLength"]], "trueShapeN":[0]})
                ], ignore_index=True)
                df_ident_line = pd.concat([
                    pd.DataFrame({"position":[0], "identifiedShapeN":[0]}),
                    df_mode[["position","identifiedShape"]].rename(columns={"identifiedShape":"identifiedShapeN"}),
                    pd.DataFrame({"position":[params["bridgeLength"]], "identifiedShapeN":[0]})
                ], ignore_index=True)

                line_true = alt.Chart(df_true_line).mark_line(
                    strokeDash=[5,3],
                    color="gray"
                ).encode(
                    x=alt.X("position", title="Position (m)"),
                    y=alt.Y("trueShapeN", title="Normalized Amplitude", scale=alt.Scale(domain=[-1.1,1.1]))
                )
                line_ident = alt.Chart(df_ident_line).mark_line(color="red").encode(
                    x="position",
                    y=alt.Y("identifiedShapeN", scale=alt.Scale(domain=[-1.1,1.1]))
                )
                points_ident = alt.Chart(df_mode).mark_point(
                    color="red",
                    filled=True,
                    size=50
                ).encode(
                    x="position",
                    y=alt.Y("identifiedShape", scale=alt.Scale(domain=[-1.1,1.1])),
                    tooltip=["sensorId", "identifiedShape"]
                )

                chart_mode = (line_true + line_ident + points_ident).properties(height=250).interactive()
                st.altair_chart(chart_mode, use_container_width=True)

                st.caption(f"**Frequency:** {float(mode_info['identifiedFrequency']):.2f} Hz "
                           f"(True: {float(mode_info['trueFrequency']):.2f} Hz), "
                           f"**MAC:** {mode_info['mac']}, "
                           f"**Error:** {mode_info['frequencyError']}%")

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
    6. Carini, M. R., & Rocha, M. M. *CESSIPy: Civil Engineer Stochastic System Identification for Python* 
       (https://github.com/MatheusCarini/CESSIPy)

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

if __name__=="__main__":
    main()
