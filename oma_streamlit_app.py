import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

try:
    # 1) Streamlit requires set_page_config() as the first command.
    st.set_page_config(
        page_title="OMA Tool with FDD & SSI",
        layout="wide"
    )
except Exception as e:
    st.error(f"Error in set_page_config: {e}")

################################################################################
# 2) General helper functions (no Streamlit calls).
################################################################################

def cpsd_fft_based(x, y, Fs):
    """
    Optional helper for cross-spectrum using direct FFT instead of scipy.signal.csd.
    """
    try:
        from scipy.fft import fft, fftfreq
        n = len(x)
        X = fft(x, n=n)
        Y = fft(y, n=n)
        Gxy = (X * np.conjugate(Y)) / n
        freq = fftfreq(n, d=1.0 / Fs)
        half = n // 2 + 1
        return freq[:half], Gxy[:half]
    except Exception as e:
        st.error(f"Error in cpsd_fft_based: {e}")
        return None, None

def calculate_mac(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the Modal Assurance Criterion (MAC) between two vectors:
       MAC = (|v1^T v2|^2) / ((v1^T v1)(v2^T v2)).
    MAC close to 1.0 indicates high similarity.
    """
    try:
        numerator = (np.sum(v1 * v2))**2
        denominator = np.sum(v1**2) * np.sum(v2**2)
        if denominator < 1e-16:
            return 0.0
        return float(numerator / denominator)
    except Exception as e:
        st.error(f"Error in calculate_mac: {e}")
        return 0.0

################################################################################
# 3) FDD Implementation
################################################################################

def FDDAuto(
    Acc,
    Fs,
    Num_Modes_Considered,
    EstimeNaturalFreqs,
    set_singular_ylim=None,
    csd_method='cpsd',
    plot_internally=False
):
    """
    Frequency Domain Decomposition (FDD):
      1) Builds cross-power spectral density (CPSD) from the input signals.
      2) Performs an SVD at each frequency line to get singular values.
      3) Identifies modes by peak-picking near estimated frequencies.
      4) Returns frequency axis, first three singular values, and mode shapes.
    """
    try:
        from scipy.signal import csd as cpsd
        from scipy.linalg import svd

        # Build the cross-power spectral matrix
        PSD = np.zeros((Acc.shape[0]//2 + 1, Acc.shape[1], Acc.shape[1]), dtype=complex)
        F   = np.zeros((Acc.shape[0]//2 + 1, Acc.shape[1], Acc.shape[1]))

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
                F[:, I, J]   = f_ij
                PSD[:, I, J] = PSD_ij

        PSD_list = []
        F_list   = []
        for k in range(PSD.shape[0]):
            PSD_list.append(PSD[k, :, :])
            F_list.append(F[k, :, :])

        # Identify the modes from PSD_list
        Frq, Phi_Real, freq_axis, s1, s2, s3 = _identifier_fdd(
            PSD_list, F_list, Num_Modes_Considered, EstimeNaturalFreqs, set_singular_ylim, plot_internally
        )
        return Frq, Phi_Real, freq_axis, s1, s2, s3
    except Exception as e:
        st.error(f"Error in FDDAuto: {e}")
        import traceback
        st.code(traceback.format_exc())
        return np.zeros(Num_Modes_Considered), np.zeros((Acc.shape[1], Num_Modes_Considered)), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)

def _identifier_fdd(PSD_list, F_list, Num_Modes_Considered,
                    EstimeNaturalFreqs, set_singular_ylim, plot_internally):
    """
    Internal subroutine for FDDAuto that:
      - SVD at each frequency line => s1, s2, s3
      - Picks peaks near each estimated frequency => identified frequencies
      - Re-SVD at each peak => identified mode shapes
      - Returns freq_axis & s1,s2,s3 for external plotting
    """
    try:
        import numpy as np
        from scipy.linalg import svd

        n_freqs = len(PSD_list)
        n_chan  = PSD_list[0].shape[0]
        s1_arr = np.zeros(n_freqs)
        s2_arr = np.zeros(n_freqs)
        s3_arr = np.zeros(n_freqs)

        # SVD at each frequency => fill s1_arr, s2_arr, s3_arr
        for i in range(n_freqs):
            mat = PSD_list[i]
            u, s, vh = svd(mat)
            if len(s)>0: s1_arr[i] = s[0]
            if len(s)>1: s2_arr[i] = s[1]
            if len(s)>2: s3_arr[i] = s[2]

        freq_arr = np.stack(F_list)[:,1,1].flatten()

        # Peak-pick near each EstimeNaturalFreq
        Fp = []
        bandwidth = 0.3
        for k in range(Num_Modes_Considered):
            f_est = EstimeNaturalFreqs[k]
            lowb = max(0, f_est-bandwidth)
            highb = f_est+ bandwidth
            subset_idx = np.where((freq_arr>=lowb)&(freq_arr<=highb))[0]
            if len(subset_idx)==0:
                best_idx = np.argmin(np.abs(freq_arr-f_est))
            else:
                best_idx = subset_idx[np.argmax(s1_arr[subset_idx])]
            Fp.append([best_idx, freq_arr[best_idx]])
        Fp = np.array(Fp)
        sr = np.argsort(Fp[:,1])
        Fp = Fp[sr,:]
        identified_freqs = Fp[:,1]

        # Re-SVD at each peak => identified shapes
        Phi_Real = np.zeros((n_chan, Num_Modes_Considered))
        for j in range(Num_Modes_Considered):
            idx_peak = int(Fp[j,0])
            mat = PSD_list[idx_peak]
            u, s, vh = svd(mat)
            phi_cpx = u[:,0]
            # Phase alignment
            Y = np.imag(phi_cpx)
            X = np.column_stack((np.real(phi_cpx), np.ones(len(phi_cpx))))
            A = np.linalg.pinv(X.T@X)@(X.T@Y)
            theta = np.arctan(A[0])
            phi_rot = phi_cpx*np.exp(-1j*theta)
            phi_rot_real = np.real(phi_rot)
            # Normalize
            phi_rot_real /= np.max(np.abs(phi_rot_real))
            Phi_Real[:, j] = phi_rot_real

        # We skip internal plotting => returning freq_arr & s1_arr, s2_arr, s3_arr
        return identified_freqs, Phi_Real, freq_arr, s1_arr, s2_arr, s3_arr
    except Exception as e:
        st.error(f"Error in _identifier_fdd: {e}")
        import traceback
        st.code(traceback.format_exc())
        # Return empty arrays of appropriate sizes
        return np.zeros(Num_Modes_Considered), np.zeros((n_chan, Num_Modes_Considered)), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)

def finalize_mode_entry_fdd(mode_number, freq_true, freq_ident,
                            shape_ident, measurements, L):
    """
    Build a single dictionary describing one identified mode:
      - modeNumber, trueFrequency, identifiedFrequency, frequencyError, mac
      - modeShapes => list of (sensorId, position, trueShape, identifiedShape)
    """
    try:
        import pandas as pd
        sensor_data = []
        for i, sens in enumerate(measurements):
            x_sens = sens["position"]
            val_true = np.sin(mode_number*np.pi*x_sens/L)
            val_ident = shape_ident[i]
            sensor_data.append({
                "sensorId": sens["sensorId"],
                "position": x_sens,
                "trueShape": val_true,
                "identifiedShape": val_ident
            })

        df = pd.DataFrame(sensor_data)
        if len(df)==0:
            return {
                "modeNumber": mode_number,
                "trueFrequency": freq_true,
                "identifiedFrequency": freq_ident,
                "frequencyError": "N/A",
                "mac": "N/A",
                "modeShapes": []
            }

        max_t = df["trueShape"].abs().max()
        max_i = df["identifiedShape"].abs().max()
        if max_t<1e-12: max_t = 1
        if max_i<1e-12: max_i = 1
        df["trueShapeN"] = df["trueShape"]/max_t
        df["identifiedShapeN"] = df["identifiedShape"]/max_i
        mac_val = calculate_mac(df["trueShapeN"].values, df["identifiedShapeN"].values)
        freq_err = 0.0
        if freq_true != 0:
            freq_err = ((freq_ident - freq_true) / freq_true) * 100.0

        # build the "modeShapes" list
        shape_list = []
        for row in df.itertuples():
            shape_list.append({
                "sensorId": row.sensorId,
                "position": row.position,
                "trueShape": row.trueShapeN,
                "identifiedShape": row.identifiedShapeN
            })

        return {
            "modeNumber": mode_number,
            "trueFrequency": freq_true,
            "identifiedFrequency": freq_ident,
            "frequencyError": f"{freq_err:.2f}",
            "mac": f"{mac_val:.4f}",
            "modeShapes": shape_list
        }
    except Exception as e:
        st.error(f"Error in finalize_mode_entry_fdd: {e}")
        import traceback
        st.code(traceback.format_exc())
        return {
            "modeNumber": mode_number,
            "trueFrequency": freq_true,
            "identifiedFrequency": freq_ident, 
            "frequencyError": "Error",
            "mac": "Error",
            "modeShapes": []
        }

def perform_fdd(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Runs FDDAuto, gathers singular values, and finalizes the 3 identified modes.
    Returns (identified_modes, svd_data).
    """
    try:
        import numpy as np
        n_sensors = len(measurements)
        n_samples = len(measurements[0]["data"])
        data_mat = np.zeros((n_samples, n_sensors))
        for i, sensor in enumerate(measurements):
            data_mat[:, i] = [pt["acceleration"] for pt in sensor["data"]]

        f1 = params["mode1"]["freq"]
        f2 = params["mode2"]["freq"]
        f3 = params["mode3"]["freq"]
        est_frqs = [f1, f2, f3]

        Frq, Phi_Real, freq_axis, s1, s2, s3 = FDDAuto(
            Acc = data_mat,
            Fs = params["samplingFreq"],
            Num_Modes_Considered = 3,
            EstimeNaturalFreqs = est_frqs,
            set_singular_ylim = None,
            csd_method = 'cpsd',
            plot_internally = False
        )

        identified = []
        L = params["bridgeLength"]
        t_frqs = [f1, f2, f3]
        for i, mode_num in enumerate([1,2,3]):
            freq_true = t_frqs[i]
            freq_ident = Frq[i]
            shape_ident = Phi_Real[:, i]
            mode_dict = finalize_mode_entry_fdd(mode_num, freq_true, freq_ident,
                                             shape_ident, measurements, L)
            identified.append(mode_dict)

        # store freq_axis, s1,s2,s3 for altair
        svd_data = []
        for i in range(len(freq_axis)):
            svd_data.append({
                "frequency": freq_axis[i],
                "sv1": s1[i],
                "sv2": s2[i],
                "sv3": s3[i]
            })

        return identified, svd_data
    except Exception as e:
        st.error(f"Error in perform_fdd: {e}")
        import traceback
        st.code(traceback.format_exc())
        return [], []

################################################################################
# 4) Real SSI Implementation
################################################################################

def real_ssi_method(output, fs, ncols, nrows, cut):
    """
    This function is adapted from a typical Subspace Identification approach, 
    parallel to the MATLAB code snippet shared.
    """
    try:
        import numpy as np
        from numpy.linalg import svd, inv, pinv, eig

        # If needed, transpose output so that (#outputs < #samples)
        outs, npts = output.shape
        if outs > npts:
            output = output.T
            outs, npts = output.shape

        # block sizes
        brows = nrows // outs
        nrows = outs * brows
        bcols = ncols // 1
        ncols = 1 * bcols
        m = 1
        q = outs

        # Build Yp, Yf
        Yp = np.zeros((nrows//2, ncols), dtype=float)
        Yf = np.zeros((nrows//2, ncols), dtype=float)
        half_brows = brows // 2

        for ii in range(1, half_brows+1):
            for jj in range(1, bcols+1):
                r_start = (ii-1) * q
                r_end = ii * q
                c_start = (jj-1) * m
                c_end = jj * m
                left = ((jj-1) + (ii-1)) * m
                right = ((jj) + (ii-1)) * m
                # Check bounds before assignment
                if left < right and right <= output.shape[1]:
                    Yp[r_start:r_end, c_start:c_end] = output[:, left:right]

        for ii in range(half_brows+1, brows+1):
            i_ = ii - half_brows
            for jj in range(1, bcols+1):
                r_start = (i_-1) * q
                r_end = i_ * q
                c_start = (jj-1) * m
                c_end = jj * m
                left = ((jj-1) + (ii-1)) * m
                right = ((jj) + (ii-1)) * m
                # Check bounds before assignment
                if left < right and right <= output.shape[1]:
                    Yf[r_start:r_end, c_start:c_end] = output[:, left:right]

        # Projection
        O = Yf @ Yp.T @ pinv(Yp @ Yp.T) @ Yp

        # SVD
        R1, Sigma1, S1 = svd(O, full_matrices=False)
        sv = Sigma1
        # Keep "cut"
        cut = min(cut, len(sv))  # Ensure cut doesn't exceed singular values length
        D = np.diag(np.sqrt(sv[:cut]))
        Rn = R1[:, :cut]
        Sn = S1[:cut, :]

        Obs = Rn @ D
        top_rows = (nrows//2) - q
        Obs_up = Obs[:top_rows, :]
        Obs_down = Obs[q:nrows//2, :]
        A = pinv(Obs_up) @ Obs_down
        C = Obs[:q, :]

        # eigen decomposition => freq, damping
        w, V = eig(A)
        Lambda = w
        # discrete => s = log(lambda)*fs
        s = np.log(Lambda) * fs
        zeta = -np.real(s) / np.abs(s) * 100.0
        fd = np.imag(s) / (2.0*np.pi)
        shapes = C @ V

        # Participation factor
        InvV = inv(V)
        partfac = np.std((InvV @ D @ Sn).T, axis=0)

        # For simplicity, define placeholders for EMAC, MPC, CMI. 
        # One can implement them if needed.
        EMAC = np.ones(cut) * 80
        MPC = np.ones(cut) * 85
        CMI = EMAC * MPC / 100.0

        # Sort ascending freq
        idx_sort = np.argsort(fd)
        fd = fd[idx_sort]
        zeta = zeta[idx_sort]
        shapes = shapes[:, idx_sort]
        partfac = partfac[idx_sort]
        EMAC = EMAC[idx_sort]
        MPC = MPC[idx_sort]
        CMI = CMI[idx_sort]

        # Remove negative freq or freq > 0.499 fs
        mask = (fd > 0) & (fd < 0.499 * fs)
        if np.any(mask):
            fd = fd[mask]
            zeta = zeta[mask]
            shapes = shapes[:, mask]
            partfac = partfac[mask]
            EMAC = EMAC[mask]
            MPC = MPC[mask]
            CMI = CMI[mask]
        else:
            # If no frequencies match the criteria, return empty arrays
            fd = np.array([])
            zeta = np.array([])
            shapes = np.zeros((q, 0))
            partfac = np.array([])
            EMAC = np.array([])
            MPC = np.array([])
            CMI = np.array([])

        # Check if we have any valid modes
        if len(fd) > 0:
            # Undamped freq => fd1 = fd/sqrt(1-(zeta/100)^2)
            fd1 = fd / np.sqrt(1 - (zeta/100.0)**2 + 1e-16)
            idx_asc = np.argsort(fd1)
            fd1 = fd1[idx_asc]
            zeta = zeta[idx_asc]
            shapes = shapes[:, idx_asc]
            partfac = partfac[idx_asc]
            EMAC = EMAC[idx_asc]
            MPC = MPC[idx_asc]
            CMI = CMI[idx_asc]

            # shape normalization
            # For each mode, rotate to make real & normalize
            n_modes = shapes.shape[1]
            ModeShapeS = np.zeros_like(shapes)
            for jj in range(n_modes):
                abscol = np.abs(shapes[:, jj])
                if np.any(abscol):  # Check if not all zeros
                    rowmax = np.argmax(abscol)
                    b = -np.angle(shapes[rowmax, jj])
                    shape_rot = shapes[:, jj] * np.exp(1j*b)
                    shape_rot = np.real(shape_rot)
                    normv = np.sqrt(np.sum(shape_rot**2))
                    if normv < 1e-12: normv = 1.0
                    shape_rot /= normv
                    ModeShapeS[:, jj] = shape_rot
            shapes = ModeShapeS
        else:
            # If no valid frequencies, set empty arrays
            fd1 = np.array([])
            n_modes = 0

        Result = {
            "Parameters":{
                "NaFreq": fd1,       # undamped freq
                "DampRatio": zeta,
                "ModeShape": shapes
            },
            "Indicators":{
                "EMAC": EMAC,
                "MPC": MPC,
                "CMI": CMI,
                "partfac": partfac
            },
            "Matrices":{
                "A": A,
                "C": C
            }
        }
        return Result
    except Exception as e:
        st.error(f"Error in real_ssi_method: {e}")
        import traceback
        st.code(traceback.format_exc())
        return {
            "Parameters": {
                "NaFreq": np.array([]),
                "DampRatio": np.array([]),
                "ModeShape": np.zeros((q, 0))
            },
            "Indicators": {
                "EMAC": np.array([]),
                "MPC": np.array([]),
                "CMI": np.array([]),
                "partfac": np.array([])
            },
            "Matrices": {
                "A": np.eye(1),
                "C": np.zeros((q, 1))
            }
        }

def perform_ssi_real(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Runs the real_ssi_method to produce identified modes from data.
    Then converts them to a format consistent with pinned-shape visualization.
    """
    try:
        import numpy as np

        # Build output array => (#sensors, #samples)
        n_sensors = len(measurements)
        n_samples = len(measurements[0]["data"])
        output = np.zeros((n_sensors, n_samples))
        for i, sensor in enumerate(measurements):
            output[i,:] = [pt["acceleration"] for pt in sensor["data"]]

        fs = params["samplingFreq"]
        # Heuristics for nrows, ncols, cut
        # These can be tuned
        nrows = 60
        ncols = max(30, n_samples//2)
        cut = 8

        # run subspace
        Result = real_ssi_method(output, fs, ncols, nrows, cut)
        # extract modes
        freqs = Result["Parameters"]["NaFreq"]    # shape (#modes,)
        zeta = Result["Parameters"]["DampRatio"]
        shapes = Result["Parameters"]["ModeShape"] # shape => (#sensors, #modes)
        
        # Check if we found any modes
        if freqs.size == 0:
            return [], []
            
        # We pick min(3, #modes)
        n_found = shapes.shape[1]
        L = params["bridgeLength"]

        # We'll attempt to produce up to 3 modes
        identified = []
        # compare with known freq if we want, or just store freq= freqs[i].
        # For clarity, we can attempt to match each i with the i-th "true freq" if i<3
        # if not enough modes => partial
        true_freqs = [
            params["mode1"]["freq"],
            params["mode2"]["freq"],
            params["mode3"]["freq"]
        ]
        n_id = min(3, n_found)
        for i in range(n_id):
            freq_ident = freqs[i]
            freq_true = true_freqs[i] if i < 3 else 0.0
            shape_vec = shapes[:, i]
            # finalize
            single_mode = finalize_mode_entry_fdd(i+1, freq_true, freq_ident, shape_vec, measurements, L)
            identified.append(single_mode)

        # no singular-value data => empty
        svd_data = []
        return identified, svd_data
    except Exception as e:
        st.error(f"Error in perform_ssi_real: {e}")
        import traceback
        st.code(traceback.format_exc())
        return [], []

################################################################################
# 5) Data Generation & Analysis Dispatcher
################################################################################

def generate_synthetic_data(params: dict):
    """
    Creates 3-mode + noise synthetic data for demonstration.
    """
    try:
        import numpy as np
        bL = params["bridgeLength"]
        n_sensors = params["numSensors"]
        mo1 = params["mode1"]
        mo2 = params["mode2"]
        mo3 = params["mode3"]
        noise = params["noiseLevel"]
        fs = params["samplingFreq"]
        dur = params["duration"]

        dt = 1.0/fs
        N = int(np.floor(dur*fs))
        positions = np.linspace(bL/(n_sensors+1), bL*(n_sensors/(n_sensors+1)), n_sensors)

        time_vec = np.arange(N) * dt
        resp1 = mo1["amp"] * np.sin(2*np.pi * mo1["freq"] * time_vec)
        resp2 = mo2["amp"] * np.sin(2*np.pi * mo2["freq"] * time_vec)
        resp3 = mo3["amp"] * np.sin(2*np.pi * mo3["freq"] * time_vec)

        shape_list = []
        for x in positions:
            s1 = np.sin(np.pi*x/bL)
            s2 = np.sin(2*np.pi*x/bL)
            s3 = np.sin(3*np.pi*x/bL)
            shape_list.append((s1,s2,s3))

        measurements = []
        for i, x in enumerate(positions):
            shape1, shape2, shape3 = shape_list[i]
            data_list = []
            for t in range(N):
                M1 = shape1 * resp1[t]
                M2 = shape2 * resp2[t]
                M3 = shape3 * resp3[t]
                noise_val = noise * (2*np.random.rand()-1)
                total_acc = M1 + M2 + M3 + noise_val
                data_list.append({
                    "time": time_vec[t],
                    "acceleration": total_acc
                })
            measurements.append({
                "sensorId": f"S{i+1}",
                "position": x,
                "data": data_list
            })
        return measurements
    except Exception as e:
        st.error(f"Error in generate_synthetic_data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return []

def perform_analysis(analysis_method: str, params: dict, measurements: List[Dict]):
    """
    Switch between FDD or real SSI. 
    Returns identified_modes, svd_data. 
    For FDD => we also produce s1,s2,s3 for plotting. 
    For SSI => no singular-value data => empty.
    """
    try:
        if analysis_method == "FDD":
            return perform_fdd(params, measurements)
        elif analysis_method == "SSI":
            return perform_ssi_real(params, measurements)
        else:
            return [], []
    except Exception as e:
        st.error(f"Error in perform_analysis: {e}")
        import traceback
        st.code(traceback.format_exc())
        return [], []

################################################################################
# 6) main() for Streamlit UI
################################################################################

def main():
    try:
        st.markdown("""
        # Operational Modal Analysis (OMA) - FDD & Real SSI
        This tool demonstrates:
        - **FDD**: Frequency Domain Decomposition with cross-power spectra & SVD.
        - **SSI**: A subspace identification approach that forms Hankel blocks, obtains system 
          matrices, then extracts frequencies, damping, and mode shapes.

        **Developer's Remarks**: 
        - Adjust parameters, generate synthetic data, and pick an analysis method. 
        - Observe how the modes compare with the ground truth. 
        """)

        with st.expander("Educational Notes", expanded=False):
            st.write("""
            **FDD** identifies modes in the frequency domain by looking at the singular values 
            of the cross-power spectral matrix. Peaks in the first singular value typically 
            reveal modes.  
            
            **SSI** (subspace) uses time-domain block Hankel matrices, SVD-based factorizations, 
            and eigen-decompositions to find state-space parameters (A, C). The eigenvalues 
            yield frequencies & damping, and the eigenvectors produce mode shapes.

            The pinned-shape plots show how the identified shape compares with a sine-based 
            'true' shape at each sensor's position. 
            """)

        # Initialize session state variables if they don't exist
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
            st.session_state.measurements = []
        if "identified_modes" not in st.session_state:
            st.session_state.identified_modes = []
        if "svd_data" not in st.session_state:
            st.session_state.svd_data = []
        if "generation_complete" not in st.session_state:
            st.session_state.generation_complete = False
        if "is_analyzing" not in st.session_state:
            st.session_state.is_analyzing = False

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
            params["mode1"]["amp"] = st.slider("Mode 1 Amplitude", 10.0, 100.0, params["mode1"]["amp"], 5.0)
            params["mode2"]["freq"] = st.slider("Mode 2 Frequency (Hz)", 5.0, 15.0, params["mode2"]["freq"], 0.5)
            params["mode2"]["amp"] = st.slider("Mode 2 Amplitude", 1.0, 50.0, params["mode2"]["amp"], 1.0)

        with col2:
            params["mode3"]["freq"] = st.slider("Mode 3 Frequency (Hz)", 15.0, 25.0, params["mode3"]["freq"], 0.5)
            params["mode3"]["amp"] = st.slider("Mode 3 Amplitude", 0.1, 10.0, params["mode3"]["amp"], 0.1)

        with st.expander("Advanced Parameters", expanded=False):
            adv1, adv2 = st.columns(2)
            with adv1:
                params["noiseLevel"] = st.slider("Noise Level", 0.0, 10.0, params["noiseLevel"], 0.5)
            with adv2:
                params["samplingFreq"] = st.slider("Sampling Frequency (Hz)", 20.0, 200.0, params["samplingFreq"], 10.0)
                params["duration"] = st.slider("Duration (seconds)", 5.0, 60.0, params["duration"], 5.0)

        st.session_state.params = params

        st.markdown("---")
        st.subheader("Data Generation")
        if st.button("Generate Synthetic Measurements"):
            with st.spinner("Generating synthetic data..."):
                st.session_state.measurements = generate_synthetic_data(st.session_state.params)
                st.session_state.identified_modes = []
                st.session_state.svd_data = []
                st.session_state.generation_complete = True

        measurements = st.session_state.measurements
        if st.session_state.generation_complete and measurements:
            st.markdown("**Synthetic Measurements Preview**")
            first_data = measurements[0]["data"][:500]  # partial data for quick view
            if first_data:
                df_sensor = pd.DataFrame(first_data)
                st.caption(f"Time History (Sensor {measurements[0]['sensorId']})")
                line_chart = alt.Chart(df_sensor).mark_line().encode(
                    x=alt.X("time", title="Time (s)"),
                    y=alt.Y("acceleration", title="Acceleration"),
                    tooltip=["time", "acceleration"]
                ).properties(height=300)
                st.altair_chart(line_chart, use_container_width=True)

            # Sensor positions
            st.caption("**Sensor Positions**")
            df_pos = pd.DataFrame({
                "sensorId": [m["sensorId"] for m in measurements],
                "position": [m["position"] for m in measurements],
                "y": [0.5] * len(measurements)
            })
            scatter_chart = alt.Chart(df_pos).mark_point(size=100, shape="triangle").encode(
                x=alt.X("position", title="Position (m)"),
                y=alt.Y("y", title="", scale=alt.Scale(domain=[0, 1])),
                tooltip=["sensorId", "position"]
            ).properties(height=150)
            st.altair_chart(scatter_chart, use_container_width=True)
            st.caption(" | ".join([f"{m['sensorId']}: {m['position']:.2f} m" for m in measurements]))

        st.markdown("---")
        st.subheader("Analysis")

        analysis_method = st.selectbox("Choose Analysis Method", ("FDD", "SSI"))
        
        def run_analysis():
            st.session_state.is_analyzing = True
            with st.spinner(f"Performing {analysis_method} analysis..."):
                identified_modes, svd_data = perform_analysis(analysis_method, st.session_state.params, st.session_state.measurements)
                st.session_state.identified_modes = identified_modes
                st.session_state.svd_data = svd_data
            st.session_state.is_analyzing = False

        st.button("Perform Analysis", on_click=run_analysis, disabled=not st.session_state.generation_complete or st.session_state.is_analyzing)

        identified_modes = st.session_state.identified_modes
        svd_data = st.session_state.svd_data
        if identified_modes:
            st.markdown(f"### Analysis Results ({analysis_method})")

            if analysis_method == "FDD" and svd_data:
                st.markdown("#### Singular Values (SV1, SV2, SV3)")
                df_svd = pd.DataFrame(svd_data)
                max_x = 2.0 * st.session_state.params["mode3"]["freq"]

                col_sv = st.columns(3)
                with col_sv[0]:
                    chart_sv1 = alt.Chart(df_svd).mark_line().encode(
                        x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                        y=alt.Y("sv1", title="Amplitude"),
                        tooltip=["frequency", "sv1"]
                    ).properties(height=300).interactive()
                    st.altair_chart(chart_sv1, use_container_width=True)
                with col_sv[1]:
                    chart_sv2 = alt.Chart(df_svd).mark_line(color="red").encode(
                        x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                        y=alt.Y("sv2", title="Amplitude"),
                        tooltip=["frequency", "sv2"]
                    ).properties(height=300).interactive()
                    st.altair_chart(chart_sv2, use_container_width=True)
                with col_sv[2]:
                    chart_sv3 = alt.Chart(df_svd).mark_line(color="green").encode(
                        x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                        y=alt.Y("sv3", title="Amplitude"),
                        tooltip=["frequency", "sv3"]
                    ).properties(height=300).interactive()
                    st.altair_chart(chart_sv3, use_container_width=True)

            st.markdown("#### Identified Modal Parameters")
            df_modes = pd.DataFrame([
                {
                    "Mode": m["modeNumber"],
                    "True Frequency (Hz)": f"{m['trueFrequency']:.2f}",
                    "Identified Frequency (Hz)": f"{m['identifiedFrequency']:.2f}",
                    "Error (%)": m["frequencyError"],
                    "MAC": m["mac"]
                }
                for m in identified_modes
            ])
            st.table(df_modes)

            st.markdown("#### Mode Shape Visualization")
            n_modes = len(identified_modes)
            col_vis = st.columns(n_modes)
            for i, mode_info in enumerate(identified_modes):
                with col_vis[i]:
                    st.markdown(f"**Mode {mode_info['modeNumber']}**")
                    df_mode = pd.DataFrame(mode_info["modeShapes"])

                    needed_cols = {"position", "trueShape", "identifiedShape"}
                    if not needed_cols.issubset(df_mode.columns) or df_mode.empty:
                        st.write("No shape data available for pinned-shape plot.")
                        continue

                    # pinned shape approach
                    df_true_line = pd.concat([
                        pd.DataFrame({"position": [0], "trueShapeN": [0]}),
                        df_mode[["position", "trueShape"]].rename(columns={"trueShape": "trueShapeN"}),
                        pd.DataFrame({"position": [params['bridgeLength']], "trueShapeN": [0]})
                    ], ignore_index=True)
                    df_ident_line = pd.concat([
                        pd.DataFrame({"position": [0], "identifiedShapeN": [0]}),
                        df_mode[["position", "identifiedShape"]].rename(columns={"identifiedShape": "identifiedShapeN"}),
                        pd.DataFrame({"position": [params['bridgeLength']], "identifiedShapeN": [0]})
                    ], ignore_index=True)

                    line_true = alt.Chart(df_true_line).mark_line(strokeDash=[5, 3], color="gray").encode(
                        x=alt.X("position", title="Position (m)"),
                        y=alt.Y("trueShapeN", title="Normalized Amplitude", scale=alt.Scale(domain=[-1.1, 1.1]))
                    )
                    line_ident = alt.Chart(df_ident_line).mark_line(color="red").encode(
                        x="position",
                        y=alt.Y("identifiedShapeN", scale=alt.Scale(domain=[-1.1, 1.1]))
                    )
                    points_ident = alt.Chart(df_mode).mark_point(color="red", filled=True, size=50).encode(
                        x="position",
                        y=alt.Y("identifiedShape", scale=alt.Scale(domain=[-1.1, 1.1])),
                        tooltip=["sensorId", "identifiedShape"]
                    )
                    chart_mode = (line_true + line_ident + points_ident).properties(height=250).interactive()
                    st.altair_chart(chart_mode, use_container_width=True)

                    st.caption(
                        f"**Freq:** {float(mode_info['identifiedFrequency']):.2f} Hz "
                        f"(True: {float(mode_info['trueFrequency']):.2f} Hz), "
                        f"**MAC:** {mode_info['mac']}, "
                        f"**Error:** {mode_info['frequencyError']}%"
                    )

        st.markdown("---")
        st.subheader("References & Closing Remarks")
        st.markdown("""
        - **FDD** references:
          1. Brincker, R., Zhang, L., & Andersen, P. (2000). *Modal Identification from Ambient Responses Using Frequency Domain Decomposition.* 
             Proceedings of the 18th International Modal Analysis Conference (IMAC).
        - **SSI** references:
          2. Peeters, B., & De Roeck, G. (2001). *Stochastic System Identification for Operational Modal Analysis: A Review.* 
             Journal of Dynamic Systems, Measurement, and Control, 123(4), 659–667.
          3. Rainieri, C., & Fabbrocino, G. (2014). *Operational Modal Analysis of Civil Engineering Structures.* Springer.
        - This code highlights how both FDD (spectral peaks + SVD) and SSI (Hankel matrix + subspace factorization) can reveal 
          frequencies, damping, and mode shapes from output-only data.

        Thank you for using this OMA educational tool!
        """)
    except Exception as e:
        st.error(f"An error occurred in the main function: {e}")
        import traceback
        st.code(traceback.format_exc())
