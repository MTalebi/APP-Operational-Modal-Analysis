import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

################################################################################
#                         FDDAuto Snippet (Modified)
################################################################################
"""
We modify FDDAuto to:
  - Accept a new parameter (e.g., plot_internally=False)
  - Skip the original Matplotlib plot in the 'Identifier' subroutine
  - Return the singular values s1, s2, s3 and frequency array for external plotting
"""

from scipy.signal import csd as cpsd
from scipy.fftpack import fft, fftfreq
from scipy.linalg import svd

st.set_page_config(
    page_title="Operational Modal Analysis Tool",
    layout="wide"
)

def cpsd_fft_based(x, y, Fs):
    n = len(x)
    X = fft(x, n=n)
    Y = fft(y, n=n)
    Gxy = (X * np.conjugate(Y)) / n
    freq = fftfreq(n, d=1.0 / Fs)
    half = n // 2 + 1
    return freq[:half], Gxy[:half]

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
    Frequency Domain Decomposition with optional internal plot turned OFF by default.
    Returns:
      - identified frequencies (array of length Num_Modes_Considered)
      - identified mode shapes (shape: n_channels x Num_Modes_Considered)
      - freq array (size n_freq_bins)
      - s1, s2, s3 arrays (size n_freq_bins) => the first 3 singular values
    """
    # Compute Cross-Power Spectral Density (PSD) matrix
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

    # Now the "Identifier" subroutine:
    Frq, Phi_Real, freq_axis, s1, s2, s3 = Identifier(
        PSD_list, 
        F_list, 
        Num_Modes_Considered, 
        EstimeNaturalFreqs, 
        set_singular_ylim, 
        plot_internally
    )

    return Frq, Phi_Real, freq_axis, s1, s2, s3

def Identifier(PSD, F, Num_Modes_Considered, EstimeNaturalFreqs, set_singular_ylim, plot_internally):
    """
    Modified to skip the original Matplotlib plotting if plot_internally=False.

    Returns:
      - NaturalIDFrq: the identified frequencies
      - Phi_Real: the identified mode shapes
      - freq: frequency array
      - s1, s2, s3: arrays of singular values
    """
    import numpy as np

    # Compute SVD of PSD at each freq
    n_freqs = len(PSD)
    n_chan = PSD[0].shape[0]
    s1_arr = np.zeros(n_freqs)
    s2_arr = np.zeros(n_freqs)
    s3_arr = np.zeros(n_freqs)

    for K in range(n_freqs):
        PSD_2DMatrix = PSD[K]
        u, s, _ = svd(PSD_2DMatrix)
        s1_arr[K] = s[0] if len(s) > 0 else 0.0
        s2_arr[K] = s[1] if len(s) > 1 else 0.0
        s3_arr[K] = s[2] if len(s) > 2 else 0.0

    freq_arr = np.stack(F)[:, 1, 1].flatten()

    # Peak picking near each EstimeNaturalFreqs
    Fp = []
    bandwidth = 0.3
    for k in range(Num_Modes_Considered):
        f_est = EstimeNaturalFreqs[k]
        lowb = f_est - bandwidth
        highb= f_est + bandwidth
        if lowb<0: lowb=0
        # subset
        subset_idx = np.where((freq_arr>=lowb) & (freq_arr<=highb))[0]
        if len(subset_idx)==0:
            best_idx = np.argmin(np.abs(freq_arr-f_est))
        else:
            # pick max of s1 in that region
            best_idx = subset_idx[np.argmax(s1_arr[subset_idx])]
        Fp.append([best_idx, freq_arr[best_idx]])

    Fp = np.array(Fp)
    sr = np.argsort(Fp[:,1])
    Fp = Fp[sr,:]
    NaturalIDFrq = Fp[:,1]

    # For each peak, do SVD again to get the mode shapes
    Phi_Real = np.zeros((n_chan, Num_Modes_Considered))
    for j in range(Num_Modes_Considered):
        peak_idx = int(Fp[j,0])
        PSD_2DMatrix = PSD[peak_idx]
        u, s, vh = svd(PSD_2DMatrix)
        phi_cpx = u[:, 0]
        # small real-imag alignment
        Y = np.imag(phi_cpx)
        X = np.column_stack((np.real(phi_cpx), np.ones(len(phi_cpx))))
        A = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
        theta = np.arctan(A[0])
        phi_rot = phi_cpx * np.exp(-1j*theta)
        phi_rot_real = np.real(phi_rot)
        phi_rot_real /= np.max(np.abs(phi_rot_real))
        Phi_Real[:, j] = phi_rot_real

    # If plot_internally=True, we do the old S1 dB plot, else skip
    if plot_internally:
        import matplotlib.pyplot as plt
        s1_db = 10.0*np.log10(s1_arr)
        fig = plt.figure(figsize=(8,5))
        ax = fig.subplots()
        ax.plot(freq_arr, s1_db, 'k-', lw=2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("1st Singular Value (dB)")
        for i in range(Fp.shape[0]):
            idxp = int(Fp[i,0])
            freq_p = Fp[i,1]
            ax.plot(freq_p, s1_db[idxp], 'ro')
        st.pyplot(fig)
        plt.close(fig)

    return NaturalIDFrq, Phi_Real, freq_arr, s1_arr, s2_arr, s3_arr


################################################################################
#                            SSI (Placeholder)
################################################################################
"""
Use a simple placeholder for SSI or your advanced method.
We'll just keep the same structure as before, returning 
some identified modes.
"""
def perform_ssi(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    # ... your real SSI code ...
    # For brevity, we do a minimal placeholder
    identified_modes = []
    svd_data = []
    # ...
    from numpy.linalg import eig
    # Example
    identified_modes = [
        {
            "modeNumber": 1,
            "trueFrequency": params["mode1"]["freq"],
            "identifiedFrequency": params["mode1"]["freq"]*1.01,
            "frequencyError": f"{1.0:.2f}",
            "mac": "0.95",
            "modeShapes": []
        },
        {
            "modeNumber": 2,
            "trueFrequency": params["mode2"]["freq"],
            "identifiedFrequency": params["mode2"]["freq"]*0.98,
            "frequencyError": f"{2.0:.2f}",
            "mac": "0.90",
            "modeShapes": []
        },
        {
            "modeNumber": 3,
            "trueFrequency": params["mode3"]["freq"],
            "identifiedFrequency": params["mode3"]["freq"]*1.02,
            "frequencyError": f"{2.0:.2f}",
            "mac": "0.93",
            "modeShapes": []
        },
    ]
    return identified_modes, svd_data


################################################################################
#                           FDD Wrapper
################################################################################
"""
perform_fdd calls FDDAuto with plot_internally=False, 
stores s1, s2, s3, freq => returns them for an external altair plot
"""
def perform_fdd(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    import numpy as np

    n_sensors = len(measurements)
    n_samples = len(measurements[0]["data"])
    data_matrix = np.zeros((n_samples, n_sensors))
    for i, sens in enumerate(measurements):
        data_matrix[:, i] = [pt["acceleration"] for pt in sens["data"]]

    mode1_f = params["mode1"]["freq"]
    mode2_f = params["mode2"]["freq"]
    mode3_f = params["mode3"]["freq"]
    estimated_frqs = [mode1_f, mode2_f, mode3_f]

    # Call FDDAuto with plot_internally=False
    Frq, Phi_Real, freq_axis, s1, s2, s3 = FDDAuto(
        Acc=data_matrix,
        Fs=params["samplingFreq"],
        Num_Modes_Considered=3,
        EstimeNaturalFreqs=estimated_frqs,
        set_singular_ylim=None,
        csd_method='cpsd',
        plot_internally=False
    )

    # Build identified_modes
    identified_modes = []
    sensor_positions = [m["position"] for m in measurements]
    true_freqs = [mode1_f, mode2_f, mode3_f]

    for i, mode_num in enumerate([1,2,3]):
        freq_true = true_freqs[i]
        freq_ident = Frq[i]
        shape_ident = Phi_Real[:, i]

        # finalize
        mode_dict = finalize_mode_entry_fdd(mode_num, freq_true, freq_ident, shape_ident, measurements, params["bridgeLength"])
        identified_modes.append(mode_dict)

    # We'll store the s1, s2, s3, freq in "svd_data"
    # so that the streamlit code can build altair plots externally
    svd_data = []
    for i in range(len(freq_axis)):
        svd_data.append({
            "frequency": freq_axis[i],
            "sv1": s1[i],
            "sv2": s2[i],
            "sv3": s3[i]
        })

    return identified_modes, svd_data

def finalize_mode_entry_fdd(mode_number, freq_true, freq_ident, shape_ident, measurements, L):
    import numpy as np
    import pandas as pd

    sensor_data = []
    for s_idx, sens in enumerate(measurements):
        x_sens = sens["position"]
        true_val = np.sin(mode_number*np.pi*x_sens/L)
        sensor_data.append({
            "sensorId": sens["sensorId"],
            "position": x_sens,
            "trueShape": true_val,
            "identifiedShape": shape_ident[s_idx]
        })

    df = pd.DataFrame(sensor_data)
    max_true = df["trueShape"].abs().max()
    max_ident= df["identifiedShape"].abs().max()
    if max_true<1e-12:
        max_true=1
    if max_ident<1e-12:
        max_ident=1
    df["trueShapeN"] = df["trueShape"]/max_true
    df["identifiedShapeN"] = df["identifiedShape"]/max_ident

    # MAC
    mac_val = calculate_mac(df["trueShapeN"].values, df["identifiedShapeN"].values)
    freq_error = ((freq_ident-freq_true)/freq_true*100.0) if freq_true!=0 else 0.0

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
        "frequencyError": f"{freq_error:.2f}",
        "mac": f"{mac_val:.4f}",
        "modeShapes": mode_shapes_list
    }

def calculate_mac(mode1: np.ndarray, mode2: np.ndarray) -> float:
    numerator = (np.sum(mode1 * mode2))**2
    denominator = np.sum(mode1**2)*np.sum(mode2**2)
    if denominator<1e-16:
        return 0.0
    return float(numerator/denominator)


################################################################################
#                      Synthetic Data Generation
################################################################################

def generate_synthetic_data(params: dict):
    import numpy as np
    bridge_length = params["bridgeLength"]
    num_sensors   = params["numSensors"]
    mode1         = params["mode1"]
    mode2         = params["mode2"]
    mode3         = params["mode3"]
    noise_level   = params["noiseLevel"]
    sampling_freq = params["samplingFreq"]
    duration      = params["duration"]

    dt = 1.0/sampling_freq
    num_samples = int(np.floor(duration*sampling_freq))

    sensor_positions = np.linspace(
        bridge_length/(num_sensors+1),
        bridge_length*(num_sensors/(num_sensors+1)),
        num_sensors
    )

    time_vector = np.arange(num_samples)*dt

    mode1_response = mode1["amp"]*np.sin(2*np.pi*mode1["freq"]*time_vector)
    mode2_response = mode2["amp"]*np.sin(2*np.pi*mode2["freq"]*time_vector)
    mode3_response = mode3["amp"]*np.sin(2*np.pi*mode3["freq"]*time_vector)

    mode_shapes = []
    for x in sensor_positions:
        s1 = np.sin(np.pi*x/bridge_length)
        s2 = np.sin(2*np.pi*x/bridge_length)
        s3 = np.sin(3*np.pi*x/bridge_length)
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
            })
        measurements.append({
            "sensorId": f"S{i+1}",
            "position": x,
            "data": data_list
        })
    return measurements


################################################################################
#                   Perform Analysis Dispatcher
################################################################################

def perform_analysis(analysis_method: str, params: dict, measurements: List[Dict]):
    if analysis_method=="FDD":
        return perform_fdd(params, measurements)
    elif analysis_method=="SSI":
        return perform_ssi(params, measurements)
    else:
        return [], []


################################################################################
#                          Streamlit App
################################################################################

def main():
    # Author info
    st.markdown("""
    **Operational Modal Analysis Interactive Learning Tool**  
    Developed by: **Mohammad Talebi-Kalaleh**  
    Contact: [talebika@ualberta.ca](mailto:talebika@ualberta.ca)  
    """)

    with st.expander("Introduction to OMA", expanded=False):
        st.write("""
        This tool demonstrates two OMA techniques:
        - **FDD**: Using your FDDAuto snippet, but we now disable internal plotting and 
          externally plot the first/second/third singular values in Altair.
        - **SSI**: A placeholder or real subspace approach.

        Adjust parameters, generate synthetic data, and compare identified modes to the known true values.
        """)

    if "params" not in st.session_state:
        st.session_state.params = {
            "bridgeLength":30.0,
            "numSensors":10,
            "mode1": {"freq":2.0, "amp":50.0},
            "mode2": {"freq":8.0, "amp":10.0},
            "mode3": {"freq":18.0, "amp":1.0},
            "noiseLevel":2.0,
            "samplingFreq":100.0,
            "duration":30.0
        }
    if "measurements" not in st.session_state:
        st.session_state["measurements"]=[]
    if "identified_modes" not in st.session_state:
        st.session_state["identified_modes"]=[]
    if "svd_data" not in st.session_state:
        st.session_state["svd_data"]=[]
    if "generation_complete" not in st.session_state:
        st.session_state["generation_complete"]=False
    if "is_analyzing" not in st.session_state:
        st.session_state["is_analyzing"]=False

    params = st.session_state.params

    st.header("Simulation Setup")

    c1, c2 = st.columns(2)
    with c1:
        params["numSensors"] = st.slider("Number of Sensors", 5,20,int(params["numSensors"]),1)
    with c2:
        params["bridgeLength"] = st.slider("Bridge Length (m)",10.0,100.0,float(params["bridgeLength"]),5.0)

    col1, col2 = st.columns(2)
    with col1:
        params["mode1"]["freq"]=st.slider("Mode 1 Frequency (Hz)",0.5,5.0,params["mode1"]["freq"],0.1)
        params["mode1"]["amp"]=st.slider("Mode 1 Amplitude",10.0,100.0,params["mode1"]["amp"],5.0)
        params["mode2"]["freq"]=st.slider("Mode 2 Frequency (Hz)",5.0,15.0,params["mode2"]["freq"],0.5)
        params["mode2"]["amp"]=st.slider("Mode 2 Amplitude",1.0,50.0,params["mode2"]["amp"],1.0)

    with col2:
        params["mode3"]["freq"]=st.slider("Mode 3 Frequency (Hz)",15.0,25.0,params["mode3"]["freq"],0.5)
        params["mode3"]["amp"]=st.slider("Mode 3 Amplitude",0.1,10.0,params["mode3"]["amp"],0.1)
    with st.expander("Advanced Parameters", expanded=False):
        a1,a2 = st.columns(2)
        with a1:
            params["noiseLevel"] = st.slider("Noise Level",0.0,10.0,params["noiseLevel"],0.5)
        with a2:
            params["samplingFreq"]=st.slider("Sampling Frequency (Hz)",20.0,200.0,params["samplingFreq"],10.0)

    st.session_state.params=params

    st.markdown("---")
    st.subheader("Data Generation")
    if st.button("Generate Synthetic Measurements"):
        st.session_state["measurements"] = generate_synthetic_data(st.session_state.params)
        st.session_state["generation_complete"]=True
        st.session_state["identified_modes"]=[]
        st.session_state["svd_data"]=[]

    measurements = st.session_state["measurements"]
    if st.session_state["generation_complete"] and measurements:
        st.markdown("**Synthetic Measurements Preview**")
        first_sensor_data = measurements[0]["data"][:500]
        if first_sensor_data:
            df_sens = pd.DataFrame(first_sensor_data)
            st.caption(f"Acceleration Time History (Sensor {measurements[0]['sensorId']})")
            line_chart= alt.Chart(df_sens).mark_line().encode(
                x=alt.X("time", title="Time (s)"),
                y=alt.Y("acceleration", title="Acceleration"),
                tooltip=["time","acceleration"]
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

        st.caption("**Sensor Positions**")
        df_pos = pd.DataFrame({
            "sensorId":[m["sensorId"] for m in measurements],
            "position":[m["position"] for m in measurements],
            "y":[0.5]*len(measurements)
        })
        scatter_chart = alt.Chart(df_pos).mark_point(size=100, shape="triangle").encode(
            x=alt.X("position",title="Position (m)"),
            y=alt.Y("y",title="",scale=alt.Scale(domain=[0,1])),
            tooltip=["sensorId","position"]
        ).properties(height=150)
        st.altair_chart(scatter_chart, use_container_width=True)

        st.caption(" | ".join([f"{m['sensorId']}: {m['position']:.2f} m" for m in measurements]))

    st.markdown("---")
    st.subheader("Analysis")
    analysis_method = st.selectbox(
        "Choose Analysis Method",
        ("FDD","SSI"),
        key="analysis_method_select"
    )

    def run_analysis():
        st.session_state["is_analyzing"]=True
        identified_modes, svd_data = perform_analysis(
            analysis_method,
            st.session_state.params,
            st.session_state["measurements"]
        )
        st.session_state["identified_modes"] = identified_modes
        st.session_state["svd_data"]        = svd_data
        st.session_state["is_analyzing"]    = False

    st.button("Perform Analysis", on_click=run_analysis, disabled=not st.session_state["generation_complete"])

    identified_modes = st.session_state["identified_modes"]
    svd_data = st.session_state["svd_data"]
    if identified_modes:
        st.markdown(f"### Analysis Results ({analysis_method})")

        # If FDD => we have s1, s2, s3 in svd_data => let's create 3 altair plots
        if analysis_method=="FDD" and svd_data:
            df_svd = pd.DataFrame(svd_data)  # columns: frequency, sv1, sv2, sv3
            # We'll do 3 columns
            st.markdown("#### Singular Values (Interactive)")

            # We limit x-range up to 2 * mode3 freq
            max_x = 2.0*params["mode3"]["freq"]

            col_fdd = st.columns(3)
            # 1) SV1
            with col_fdd[0]:
                chart_sv1 = alt.Chart(df_svd).mark_line().encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv1", title="Amplitude"),
                    tooltip=["frequency","sv1"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv1, use_container_width=True)

            # 2) SV2
            with col_fdd[1]:
                chart_sv2 = alt.Chart(df_svd).mark_line(color="red").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv2", title="Amplitude"),
                    tooltip=["frequency","sv2"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv2, use_container_width=True)

            # 3) SV3
            with col_fdd[2]:
                chart_sv3 = alt.Chart(df_svd).mark_line(color="green").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv3", title="Amplitude"),
                    tooltip=["frequency","sv3"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv3, use_container_width=True)

        # show table
        st.markdown("#### Identified Modal Parameters")
        df_modes = pd.DataFrame([
            {
                "Mode":m["modeNumber"],
                "True Frequency (Hz)":f"{m['trueFrequency']:.2f}",
                "Identified Frequency (Hz)":f"{m['identifiedFrequency']:.2f}",
                "Error (%)":m["frequencyError"],
                "MAC":m["mac"]
            } for m in identified_modes
        ])
        st.table(df_modes)

        # mode shapes
        st.markdown("#### Mode Shape Visualization")
        n_modes = len(identified_modes)
        mode_cols = st.columns(n_modes)
        for i, mode_info in enumerate(identified_modes):
            with mode_cols[i]:
                st.markdown(f"**Mode {mode_info['modeNumber']}**")
                df_mode = pd.DataFrame(mode_info["modeShapes"])
                # Insert zero endpoints
                df_true_line = pd.concat([
                    pd.DataFrame({"position":[0], "trueShapeN":[0]}),
                    df_mode[["position","trueShape"]].rename(columns={"trueShape":"trueShapeN"}),
                    pd.DataFrame({"position":[params["bridgeLength"]],"trueShapeN":[0]})
                ], ignore_index=True)
                df_ident_line = pd.concat([
                    pd.DataFrame({"position":[0],"identifiedShapeN":[0]}),
                    df_mode[["position","identifiedShape"]].rename(columns={"identifiedShape":"identifiedShapeN"}),
                    pd.DataFrame({"position":[params["bridgeLength"]],"identifiedShapeN":[0]})
                ], ignore_index=True)

                line_true = alt.Chart(df_true_line).mark_line(strokeDash=[5,3], color="gray").encode(
                    x=alt.X("position", title="Position (m)"),
                    y=alt.Y("trueShapeN", title="Normalized Amplitude", scale=alt.Scale(domain=[-1.1,1.1]))
                )
                line_ident = alt.Chart(df_ident_line).mark_line(color="red").encode(
                    x="position",
                    y=alt.Y("identifiedShapeN", scale=alt.Scale(domain=[-1.1,1.1]))
                )
                points_ident = alt.Chart(df_mode).mark_point(color="red", filled=True, size=50).encode(
                    x="position",
                    y=alt.Y("identifiedShape", scale=alt.Scale(domain=[-1.1,1.1])),
                    tooltip=["sensorId","identifiedShape"]
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
    st.subheader("References & Notes")
    st.markdown("""
    - For advanced usage, consider customizing the subspace approach, 
      or adding a stabilization diagram for SSI.
    - FDD now returns three singular values: SV1, SV2, SV3, 
      plotted side by side in interactive Altair charts.
    """)

if __name__=="__main__":
    main()
