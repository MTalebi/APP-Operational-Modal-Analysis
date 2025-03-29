import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

###############################################################################
# 1) MUST call set_page_config() FIRST to avoid StreamlitSetPageConfigMustBeFirstCommandError
###############################################################################
st.set_page_config(
    page_title="Operational Modal Analysis Tool",
    layout="wide"
)

###############################################################################
# 2) Define all functions/classes that do NOT call st.* here
###############################################################################

def cpsd_fft_based(x, y, Fs):
    """
    Optional helper for cross-spectrum via FFT if not using scipy.signal.csd.
    """
    from scipy.fft import fft, fftfreq
    n = len(x)
    X = fft(x, n=n)
    Y = fft(y, n=n)
    Gxy = (X * np.conjugate(Y)) / n
    freq = fftfreq(n, d=1.0 / Fs)
    half = n // 2 + 1
    return freq[:half], Gxy[:half]

def calculate_mac(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Modal Assurance Criterion.
    """
    numerator = (np.sum(v1 * v2)) ** 2
    denominator = np.sum(v1 ** 2) * np.sum(v2 ** 2)
    if denominator < 1e-16:
        return 0.0
    return float(numerator / denominator)


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
    Frequency Domain Decomposition that returns:
      (identified_freqs, identified_shapes, freq_axis, s1, s2, s3)
    so we can externally plot s1/s2/s3 in Streamlit. 
    """
    import numpy as np
    from scipy.signal import csd as cpsd
    from scipy.linalg import svd

    # Build cross-power spectral matrix
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

    # Convert PSD slices to lists
    PSD_list = []
    F_list   = []
    for k in range(PSD.shape[0]):
        PSD_list.append(PSD[k,:,:])
        F_list.append(F[k,:,:])

    # Identify
    Frq, Phi_Real, freq_axis, s1, s2, s3 = _identifier(
        PSD_list, F_list, Num_Modes_Considered, EstimeNaturalFreqs, set_singular_ylim, plot_internally
    )
    return Frq, Phi_Real, freq_axis, s1, s2, s3

def _identifier(
    PSD_list,
    F_list,
    Num_Modes_Considered,
    EstimeNaturalFreqs,
    set_singular_ylim,
    plot_internally
):
    """
    Internal ID step:
      - SVD at each freq => s1, s2, s3
      - peak-pick near each EstimeNaturalFreq
      - re-SVD at peak => mode shapes
    """
    import numpy as np
    from scipy.linalg import svd

    n_freqs = len(PSD_list)
    n_chan  = PSD_list[0].shape[0]
    s1_arr = np.zeros(n_freqs)
    s2_arr = np.zeros(n_freqs)
    s3_arr = np.zeros(n_freqs)

    # SVD at each frequency line
    for i in range(n_freqs):
        mat = PSD_list[i]
        u, s, vh = svd(mat)
        if len(s)>0: s1_arr[i] = s[0]
        if len(s)>1: s2_arr[i] = s[1]
        if len(s)>2: s3_arr[i] = s[2]

    freq_arr = np.stack(F_list)[:,1,1].flatten()

    # Peak picking near EstimeNaturalFreqs
    Fp = []
    bandwidth = 0.3
    for k in range(Num_Modes_Considered):
        f_est = EstimeNaturalFreqs[k]
        lowb = max(f_est - bandwidth, 0)
        highb= f_est + bandwidth
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
    # For each peak, do SVD again => mode shapes
    Phi_Real = np.zeros((n_chan, Num_Modes_Considered))
    for j in range(Num_Modes_Considered):
        idx_peak = int(Fp[j,0])
        mat = PSD_list[idx_peak]
        u, s, vh = svd(mat)
        phi_cpx = u[:,0]
        Y = np.imag(phi_cpx)
        X = np.column_stack((np.real(phi_cpx), np.ones(len(phi_cpx))))
        A = np.linalg.pinv(X.T@X)@(X.T@Y)
        theta = np.arctan(A[0])
        phi_rot = phi_cpx*np.exp(-1j*theta)
        phi_rot_real = np.real(phi_rot)
        phi_rot_real/= np.max(np.abs(phi_rot_real))
        Phi_Real[:, j] = phi_rot_real

    # We skip the "plot_internally" part; 
    # returning freq_arr, s1_arr, etc. for external plotting
    return identified_freqs, Phi_Real, freq_arr, s1_arr, s2_arr, s3_arr


def generate_synthetic_data(params: dict):
    """
    Synthetic data: 3 modes + noise. 
    Returns list of dicts: { "sensorId", "position", "data":[{"time","acceleration"},...] }
    """
    import numpy as np
    bridge_length = params["bridgeLength"]
    num_sensors   = params["numSensors"]
    m1, m2, m3    = params["mode1"], params["mode2"], params["mode3"]
    noise_level   = params["noiseLevel"]
    fs            = params["samplingFreq"]
    duration      = params["duration"]

    dt = 1.0/fs
    num_samples = int(np.floor(duration*fs))

    sensor_positions = np.linspace(
        bridge_length/(num_sensors+1),
        bridge_length*(num_sensors/(num_sensors+1)),
        num_sensors
    )
    time_vec = np.arange(num_samples)*dt

    resp1 = m1["amp"]*np.sin(2*np.pi*m1["freq"]*time_vec)
    resp2 = m2["amp"]*np.sin(2*np.pi*m2["freq"]*time_vec)
    resp3 = m3["amp"]*np.sin(2*np.pi*m3["freq"]*time_vec)

    mode_shapes = []
    for x in sensor_positions:
        s1 = np.sin(np.pi*x/bridge_length)
        s2 = np.sin(2*np.pi*x/bridge_length)
        s3 = np.sin(3*np.pi*x/bridge_length)
        mode_shapes.append((s1, s2, s3))

    measurements=[]
    for i,x in enumerate(sensor_positions):
        shape1, shape2, shape3 = mode_shapes[i]
        data_list=[]
        for t_idx in range(num_samples):
            m_1 = shape1*resp1[t_idx]
            m_2 = shape2*resp2[t_idx]
            m_3 = shape3*resp3[t_idx]
            noise = noise_level*(2*np.random.rand()-1)
            total_acc = m_1+m_2+m_3+noise
            data_list.append({
                "time":time_vec[t_idx],
                "acceleration":total_acc
            })
        measurements.append({
            "sensorId": f"S{i+1}",
            "position": x,
            "data": data_list
        })
    return measurements


def finalize_mode_entry_fdd(mode_number, freq_true, freq_ident, shape_ident, measurements, L):
    """
    Build a dictionary with (sensorId, position, trueShape, identifiedShape) + MAC + freq error
    """
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
    if len(df)==0:
        return {
            "modeNumber": mode_number,
            "trueFrequency": freq_true,
            "identifiedFrequency": freq_ident,
            "frequencyError": "N/A",
            "mac": "N/A",
            "modeShapes": []
        }

    max_true = df["trueShape"].abs().max()
    max_ident= df["identifiedShape"].abs().max()
    if max_true<1e-12: max_true=1
    if max_ident<1e-12: max_ident=1
    df["trueShapeN"] = df["trueShape"]/max_true
    df["identifiedShapeN"] = df["identifiedShape"]/max_ident
    mac_val = calculate_mac(df["trueShapeN"].values, df["identifiedShapeN"].values)
    freq_err = 0.0
    if freq_true!=0:
        freq_err = ((freq_ident-freq_true)/freq_true)*100.0

    # build modeShapes
    mode_shapes_list=[]
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
        "frequencyError": f"{freq_err:.2f}",
        "mac": f"{mac_val:.4f}",
        "modeShapes": mode_shapes_list
    }

def perform_fdd(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Calls FDDAuto, extracts s1, s2, s3 for altair plotting,
    finalizes mode shapes => identified_modes,
    returns (identified_modes, svd_data).
    """
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

    Frq, Phi_Real, freq_axis, s1, s2, s3 = FDDAuto(
        Acc=data_matrix,
        Fs=params["samplingFreq"],
        Num_Modes_Considered=3,
        EstimeNaturalFreqs=estimated_frqs,
        set_singular_ylim=None,
        csd_method='cpsd',
        plot_internally=False
    )

    # Build identified modes
    identified_modes=[]
    L = params["bridgeLength"]
    true_freqs = [mode1_f, mode2_f, mode3_f]
    for i, mode_num in enumerate([1,2,3]):
        freq_true = true_freqs[i]
        freq_ident= Frq[i]
        shape_ident = Phi_Real[:, i]
        mode_dict = finalize_mode_entry_fdd(mode_num, freq_true, freq_ident, shape_ident, measurements, L)
        identified_modes.append(mode_dict)

    # Build SVD data array
    svd_data=[]
    for i in range(len(freq_axis)):
        svd_data.append({
            "frequency": freq_axis[i],
            "sv1": s1[i],
            "sv2": s2[i],
            "sv3": s3[i]
        })

    return identified_modes, svd_data


def perform_ssi(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Placeholder or simplified SSI with empty 'modeShapes' => thus no 'trueShape' columns => skip shape plotting
    """
    # Just demonstration
    identified_modes = [
        {
            "modeNumber":1,
            "trueFrequency": params["mode1"]["freq"],
            "identifiedFrequency": params["mode1"]["freq"]*1.01,
            "frequencyError": f"{1.0:.2f}",
            "mac": "0.95",
            "modeShapes": []  # empty => no "trueShape" => skip pinned shape
        },
        {
            "modeNumber":2,
            "trueFrequency": params["mode2"]["freq"],
            "identifiedFrequency": params["mode2"]["freq"]*0.98,
            "frequencyError": f"{2.0:.2f}",
            "mac": "0.90",
            "modeShapes": []
        },
        {
            "modeNumber":3,
            "trueFrequency": params["mode3"]["freq"],
            "identifiedFrequency": params["mode3"]["freq"]*1.02,
            "frequencyError": f"{2.0:.2f}",
            "mac": "0.93",
            "modeShapes": []
        },
    ]
    return identified_modes, []


def perform_analysis(analysis_method: str, params: dict, measurements: List[Dict]):
    """
    Switch between FDD or SSI
    """
    if analysis_method=="FDD":
        return perform_fdd(params, measurements)
    elif analysis_method=="SSI":
        return perform_ssi(params, measurements)
    else:
        return [], []


###############################################################################
# 3) main() uses st.* commands
###############################################################################
def main():
    st.markdown("""
    **Operational Modal Analysis Tool**  
    _Developed by: Mohammad Talebi-Kalaleh_  
    Contact: [talebika@ualberta.ca](mailto:talebika@ualberta.ca)
    """)

    with st.expander("Introduction", expanded=False):
        st.write("""
        This app demonstrates:
        - Frequency Domain Decomposition (FDD) plotting the first 3 singular values in Altair
        - A placeholder SSI approach returning no shapes => skipping shape plot
        """)

    # Setup session-state
    if "params" not in st.session_state:
        st.session_state.params = {
            "bridgeLength": 30.0,
            "numSensors": 10,
            "mode1": {"freq":2.0, "amp":50.0},
            "mode2": {"freq":8.0, "amp":10.0},
            "mode3": {"freq":18.0, "amp":1.0},
            "noiseLevel": 2.0,
            "samplingFreq":100.0,
            "duration": 30.0
        }
    if "measurements" not in st.session_state:
        st.session_state.measurements=[]
    if "identified_modes" not in st.session_state:
        st.session_state.identified_modes=[]
    if "svd_data" not in st.session_state:
        st.session_state.svd_data=[]
    if "generation_complete" not in st.session_state:
        st.session_state.generation_complete=False
    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing=False

    params = st.session_state.params

    st.header("Simulation Setup")
    c1, c2 = st.columns(2)
    with c1:
        params["numSensors"] = st.slider("Number of Sensors",5,20,int(params["numSensors"]),1)
    with c2:
        params["bridgeLength"] = st.slider("Bridge Length (m)",10.0,100.0,float(params["bridgeLength"]),5.0)

    col1, col2 = st.columns(2)
    with col1:
        params["mode1"]["freq"] = st.slider("Mode 1 Frequency (Hz)",0.5,5.0,params["mode1"]["freq"],0.1)
        params["mode1"]["amp"]  = st.slider("Mode 1 Amplitude",10.0,100.0,params["mode1"]["amp"],5.0)
        params["mode2"]["freq"] = st.slider("Mode 2 Frequency (Hz)",5.0,15.0,params["mode2"]["freq"],0.5)
        params["mode2"]["amp"]  = st.slider("Mode 2 Amplitude",1.0,50.0,params["mode2"]["amp"],1.0)
    with col2:
        params["mode3"]["freq"] = st.slider("Mode 3 Frequency (Hz)",15.0,25.0,params["mode3"]["freq"],0.5)
        params["mode3"]["amp"]  = st.slider("Mode 3 Amplitude",0.1,10.0,params["mode3"]["amp"],0.1)

    with st.expander("Advanced Parameters", expanded=False):
        a1,a2 = st.columns(2)
        with a1:
            params["noiseLevel"]=st.slider("Noise Level",0.0,10.0,params["noiseLevel"],0.5)
        with a2:
            params["samplingFreq"]=st.slider("Sampling Frequency (Hz)",20.0,200.0,params["samplingFreq"],10.0)

    st.session_state.params=params

    st.markdown("---")
    st.subheader("Data Generation")
    if st.button("Generate Synthetic Measurements"):
        st.session_state.measurements = generate_synthetic_data(st.session_state.params)
        st.session_state.identified_modes=[]
        st.session_state.svd_data=[]
        st.session_state.generation_complete=True

    measurements = st.session_state.measurements
    if st.session_state.generation_complete and measurements:
        st.markdown("**Synthetic Measurements Preview**")
        first_data = measurements[0]["data"][:500]
        if first_data:
            df_sens = pd.DataFrame(first_data)
            st.caption(f"Time History (Sensor {measurements[0]['sensorId']})")
            line_chart = alt.Chart(df_sens).mark_line().encode(
                x=alt.X("time", title="Time (s)"),
                y=alt.Y("acceleration", title="Acceleration"),
                tooltip=["time","acceleration"]
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

        # Sensor positions
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
        st.altair_chart(scatter_chart,use_container_width=True)
        st.caption(" | ".join([f"{m['sensorId']}: {m['position']:.2f} m" for m in measurements]))

    st.markdown("---")
    st.subheader("Analysis")
    analysis_method = st.selectbox("Choose Analysis Method",("FDD","SSI"))

    def run_analysis():
        st.session_state.is_analyzing=True
        identified, svdata = perform_analysis(
            analysis_method,
            st.session_state.params,
            st.session_state.measurements
        )
        st.session_state.identified_modes = identified
        st.session_state.svd_data        = svdata
        st.session_state.is_analyzing=False

    st.button("Perform Analysis", on_click=run_analysis, disabled=not st.session_state.generation_complete)

    identified_modes = st.session_state.identified_modes
    svd_data = st.session_state.svd_data
    if identified_modes:
        st.markdown(f"### Analysis Results ({analysis_method})")

        # If FDD => we have s1, s2, s3 => plot them
        if analysis_method=="FDD" and svd_data:
            st.markdown("#### Singular Values (SV1, SV2, SV3)")
            df_svd = pd.DataFrame(svd_data)  # freq, sv1, sv2, sv3
            max_x = 2.0*params["mode3"]["freq"]

            col_sv = st.columns(3)
            with col_sv[0]:
                chart_sv1 = alt.Chart(df_svd).mark_line().encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv1", title="Amplitude"),
                    tooltip=["frequency","sv1"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv1, use_container_width=True)

            with col_sv[1]:
                chart_sv2 = alt.Chart(df_svd).mark_line(color="red").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv2", title="Amplitude"),
                    tooltip=["frequency","sv2"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv2, use_container_width=True)

            with col_sv[2]:
                chart_sv3 = alt.Chart(df_svd).mark_line(color="green").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv3", title="Amplitude"),
                    tooltip=["frequency","sv3"]
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
        col_modes = st.columns(n_modes)

        for i, mode_info in enumerate(identified_modes):
            with col_modes[i]:
                st.markdown(f"**Mode {mode_info['modeNumber']}**")
                df_mode = pd.DataFrame(mode_info["modeShapes"])

                #  ---- NEW FIX: Skip if "trueShape" or "identifiedShape" not in columns
                needed_cols = {"position","trueShape","identifiedShape"}
                if not needed_cols.issubset(df_mode.columns):
                    st.write("No shape data to display for this mode.")
                    continue
                if len(df_mode)==0:
                    st.write("Empty shape data.")
                    continue
                # Now we can do pinned shape safely
                df_true_line = pd.concat([
                    pd.DataFrame({"position":[0],"trueShapeN":[0]}),
                    df_mode[["position","trueShape"]].rename(columns={"trueShape":"trueShapeN"}),
                    pd.DataFrame({"position":[params["bridgeLength"]],"trueShapeN":[0]})
                ], ignore_index=True)
                df_ident_line = pd.concat([
                    pd.DataFrame({"position":[0],"identifiedShapeN":[0]}),
                    df_mode[["position","identifiedShape"]].rename(columns={"identifiedShape":"identifiedShapeN"}),
                    pd.DataFrame({"position":[params["bridgeLength"]],"identifiedShapeN":[0]})
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
                points_ident = alt.Chart(df_mode).mark_point(color="red", filled=True, size=50).encode(
                    x="position",
                    y=alt.Y("identifiedShape", scale=alt.Scale(domain=[-1.1,1.1])),
                    tooltip=["sensorId","identifiedShape"]
                )
                chart_mode = (line_true + line_ident + points_ident).properties(height=250).interactive()
                st.altair_chart(chart_mode,use_container_width=True)

                st.caption(
                    f"**Freq:** {float(mode_info['identifiedFrequency']):.2f} Hz "
                    f"(True: {float(mode_info['trueFrequency']):.2f} Hz), "
                    f"**MAC:** {mode_info['mac']}, "
                    f"**Error:** {mode_info['frequencyError']}%"
                )

    st.markdown("---")
    st.subheader("References & Notes")
    st.markdown("""
    **KeyError Fix**: We skip shape plotting if `"trueShape"` or `"identifiedShape"` are absent 
    (which often happens in placeholder SSI modes with empty modeShapes).

    **FDD**: Plots first/second/third singular values side-by-side in interactive Altair charts.  
    **SSI**: Placeholder that returns empty `modeShapes`.
    """)

###############################################################################
# 4) If invoked directly, run main()
###############################################################################
if __name__=="__main__":
    main()
