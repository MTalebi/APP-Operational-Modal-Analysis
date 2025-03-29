import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

###############################################################################
# 1) Must call set_page_config() first in Streamlit
###############################################################################
st.set_page_config(
    page_title="Operational Modal Analysis Tool",
    layout="wide"
)

###############################################################################
# 2) Define functions/classes that do not call st.* 
###############################################################################

def cpsd_fft_based(x, y, Fs):
    """
    Optional helper for cross-spectrum if not using scipy.signal.csd.
    """
    from scipy.fft import fft, fftfreq
    n = len(x)
    X = fft(x, n=n)
    Y = fft(y, n=n)
    Gxy = (X * np.conjugate(Y)) / n
    freq = fftfreq(n, d=1.0 / Fs)
    half = n // 2 + 1
    return freq[:half], Gxy[:half]


def calculate_mac(v1: np.ndarray, v2: np.ndarray)->float:
    """
    Modal Assurance Criterion (MAC).
    """
    numerator = (np.sum(v1 * v2))**2
    denominator = np.sum(v1**2)*np.sum(v2**2)
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
    Frequency Domain Decomposition function that:
      1) builds cross-power spectral density,
      2) SVD at each frequency => s1, s2, s3,
      3) picks peaks near EstimeNaturalFreqs,
      4) returns freq axis + s1, s2, s3 for external plotting 
    """
    from scipy.signal import csd as cpsd
    from scipy.linalg import svd

    PSD = np.zeros((Acc.shape[0]//2 +1, Acc.shape[1], Acc.shape[1]), dtype=complex)
    F   = np.zeros((Acc.shape[0]//2 +1, Acc.shape[1], Acc.shape[1]))

    # Build PSD
    for I in range(Acc.shape[1]):
        for J in range(Acc.shape[1]):
            if csd_method=='cpsd':
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
    Internal step for FDDAuto: 
      - SVD at each frequency => s1, s2, s3
      - peak picking near EstimeNaturalFreqs => identified modes
      - SVD again at each peak => mode shape
    """
    import numpy as np
    from scipy.linalg import svd

    n_freqs = len(PSD_list)
    n_chan  = PSD_list[0].shape[0]
    s1_arr = np.zeros(n_freqs)
    s2_arr = np.zeros(n_freqs)
    s3_arr = np.zeros(n_freqs)

    # SVD at each freq line
    for i in range(n_freqs):
        mat = PSD_list[i]
        u, s, vh = svd(mat)
        if len(s)>0: s1_arr[i] = s[0]
        if len(s)>1: s2_arr[i] = s[1]
        if len(s)>2: s3_arr[i] = s[2]

    freq_arr = np.stack(F_list)[:,1,1].flatten()

    # peak picking
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
    # get mode shapes by re-SVD at each peak
    Phi_Real = np.zeros((n_chan, Num_Modes_Considered))
    from scipy.linalg import svd
    for j in range(Num_Modes_Considered):
        idx_peak = int(Fp[j,0])
        mat = PSD_list[idx_peak]
        u, s, vh = svd(mat)
        phi_cpx = u[:, 0]
        # align phase
        Y = np.imag(phi_cpx)
        X = np.column_stack((np.real(phi_cpx), np.ones(len(phi_cpx))))
        A = np.linalg.pinv(X.T@X)@(X.T@Y)
        theta = np.arctan(A[0])
        phi_rot = phi_cpx*np.exp(-1j*theta)
        phi_rot_real = np.real(phi_rot)
        phi_rot_real /= np.max(np.abs(phi_rot_real))
        Phi_Real[:, j] = phi_rot_real

    # skip internal plot, returning freq_arr, s1_arr, s2_arr, s3_arr
    return identified_freqs, Phi_Real, freq_arr, s1_arr, s2_arr, s3_arr


def generate_synthetic_data(params: dict):
    """
    Synthetic data: 3 modes + noise, 
    returning a list of sensor dicts => { sensorId, position, data[] }.
    """
    import numpy as np
    bL = params["bridgeLength"]
    num_sensors = params["numSensors"]
    m1 = params["mode1"]
    m2 = params["mode2"]
    m3 = params["mode3"]
    noise_lvl = params["noiseLevel"]
    fs        = params["samplingFreq"]
    duration  = params["duration"]

    dt = 1.0/fs
    N  = int(np.floor(duration*fs))
    sensor_pos = np.linspace(
        bL/(num_sensors+1),
        bL*(num_sensors/(num_sensors+1)),
        num_sensors
    )

    time_vec = np.arange(N)*dt
    resp1 = m1["amp"]*np.sin(2*np.pi*m1["freq"]*time_vec)
    resp2 = m2["amp"]*np.sin(2*np.pi*m2["freq"]*time_vec)
    resp3 = m3["amp"]*np.sin(2*np.pi*m3["freq"]*time_vec)

    # sensor mode shapes
    mode_shapes = []
    for x in sensor_pos:
        s1 = np.sin(np.pi*x/bL)
        s2 = np.sin(2*np.pi*x/bL)
        s3 = np.sin(3*np.pi*x/bL)
        mode_shapes.append((s1,s2,s3))

    measurements=[]
    for i,x in enumerate(sensor_pos):
        shape1, shape2, shape3 = mode_shapes[i]
        data_list=[]
        for t in range(N):
            M1 = shape1*resp1[t]
            M2 = shape2*resp2[t]
            M3 = shape3*resp3[t]
            noise = noise_lvl*(2*np.random.rand()-1)
            total = M1+M2+M3+noise
            data_list.append({
                "time": time_vec[t],
                "acceleration": total
            })
        measurements.append({
            "sensorId": f"S{i+1}",
            "position": x,
            "data": data_list
        })
    return measurements


def finalize_mode_entry_fdd(mode_number, freq_true, freq_ident, shape_ident, measurements, L):
    """
    Build a single identified mode entry, including 'modeShapes' array with
    (position, trueShape, identifiedShape).
    """
    import pandas as pd
    sensor_data=[]
    for s_idx, sens in enumerate(measurements):
        x_sens = sens["position"]
        val_true = np.sin(mode_number*np.pi*x_sens/L)
        val_ident= shape_ident[s_idx]
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
    max_true = df["trueShape"].abs().max()
    max_ident= df["identifiedShape"].abs().max()
    if max_true<1e-12: max_true=1
    if max_ident<1e-12: max_ident=1
    df["trueShapeN"] = df["trueShape"]/max_true
    df["identifiedShapeN"]= df["identifiedShape"]/max_ident
    mac_val = calculate_mac(df["trueShapeN"].values, df["identifiedShapeN"].values)
    freq_err=0.0
    if freq_true!=0:
        freq_err=((freq_ident-freq_true)/freq_true)*100.0

    # build the "modeShapes" array
    modeShapes=[]
    for row in df.itertuples():
        modeShapes.append({
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
        "modeShapes": modeShapes
    }

def perform_fdd(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Calls FDDAuto => obtains freq, s1,s2,s3 => altair plotting
    Then finalize 3 modes 
    """
    import numpy as np

    n_sensors = len(measurements)
    n_samples = len(measurements[0]["data"])
    data_mat = np.zeros((n_samples, n_sensors))
    for i, sens in enumerate(measurements):
        data_mat[:, i] = [pt["acceleration"] for pt in sens["data"]]

    mode1_f = params["mode1"]["freq"]
    mode2_f = params["mode2"]["freq"]
    mode3_f = params["mode3"]["freq"]
    est_frqs= [mode1_f, mode2_f, mode3_f]

    Frq, Phi_Real, freq_axis, s1, s2, s3 = FDDAuto(
        Acc=data_mat,
        Fs=params["samplingFreq"],
        Num_Modes_Considered=3,
        EstimeNaturalFreqs=est_frqs,
        set_singular_ylim=None,
        csd_method='cpsd',
        plot_internally=False
    )

    identified_modes=[]
    L = params["bridgeLength"]
    tfreqs = [mode1_f, mode2_f, mode3_f]
    for i, mode_num in enumerate([1,2,3]):
        freq_true = tfreqs[i]
        freq_ident= Frq[i]
        shape_ident= Phi_Real[:, i]
        entry = finalize_mode_entry_fdd(mode_num, freq_true, freq_ident, shape_ident, measurements, L)
        identified_modes.append(entry)

    # store s1,s2,s3 in a list
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
    Example SSI that returns actual shapes so we can visualize them 
    and avoid KeyError. We'll mimic shape generation similarly to FDD.

    Steps:
      - We generate 3 modes 
      - For each mode: pick freq ~ true freq * small factor
        build shape ident ~ same or random
      - finalize_mode_entry_fdd => so columns match
    """
    import numpy as np

    L = params["bridgeLength"]
    n_sensors = len(measurements)

    # For demonstration, pick freq with small offset
    freq1 = params["mode1"]["freq"]*(1+0.01*(np.random.rand()-0.5))
    freq2 = params["mode2"]["freq"]*(1+0.02*(np.random.rand()-0.5))
    freq3 = params["mode3"]["freq"]*(1+0.03*(np.random.rand()-0.5))
    identified_freqs = [freq1, freq2, freq3]
    true_freqs = [
        params["mode1"]["freq"],
        params["mode2"]["freq"],
        params["mode3"]["freq"]
    ]

    # Build identified shapes. 
    # For demonstration, let's do it similarly to FDD 
    # or simply "the same shape but with small random scaling".
    # shape i = sin(i pi x / L)*some random factor
    # We'll do i in [1,2,3].
    identified_modes=[]
    for i, mode_num in enumerate([1,2,3]):
        freq_true = true_freqs[i]
        freq_ident= identified_freqs[i]

        # build a shape array matching # sensors
        # shape = sin, plus small random factor
        shape_array = []
        for s_idx, sensor in enumerate(measurements):
            x_sens = sensor["position"]
            base_val = np.sin(mode_num*np.pi*x_sens/L)
            factor   = 1.0 + 0.10*(np.random.rand()-0.5)  # +/- 5%
            ident_val= base_val*factor
            shape_array.append(ident_val)

        # finalize
        entry = finalize_mode_entry_fdd(mode_num, freq_true, freq_ident, shape_array, measurements, L)
        identified_modes.append(entry)

    # For SSI we do not produce "svd_data" for singular-value plots 
    # So we return an empty array or None
    svd_data=[]
    return identified_modes, svd_data


def perform_analysis(analysis_method: str, params: dict, measurements: List[Dict]):
    """
    Decide if we do FDD or SSI
    """
    if analysis_method=="FDD":
        return perform_fdd(params, measurements)
    elif analysis_method=="SSI":
        return perform_ssi(params, measurements)
    else:
        return [], []


###############################################################################
# 3) main() with st.* calls
###############################################################################
def main():
    st.markdown("""
    # Operational Modal Analysis Tool
    **Developed by:** Mohammad Talebi-Kalaleh  
    Contact: [talebika@ualberta.ca](mailto:talebika@ualberta.ca)

    ---
    """)

    with st.expander("Introduction", expanded=False):
        st.write("""
        This application demonstrates two OMA methods:
        1. **FDD (Frequency Domain Decomposition)** with external plotting of the first three singular values.
        2. **SSI (Stochastic Subspace Identification)** with a placeholder approach that *does* produce
           mode shape data to avoid KeyError.

        1. Adjust parameters below, 
        2. Generate synthetic data,
        3. Choose method (FDD or SSI),
        4. Perform analysis,
        5. Compare identified modes with ground-truth.
        """)

    # Session-state
    if "params" not in st.session_state:
        st.session_state.params = {
            "bridgeLength": 30.0,
            "numSensors": 10,
            "mode1": {"freq":2.0, "amp":50.0},
            "mode2": {"freq":8.0, "amp":10.0},
            "mode3": {"freq":18.0, "amp":1.0},
            "noiseLevel": 2.0,
            "samplingFreq":100.0,
            "duration":30.0
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
            df_sensor = pd.DataFrame(first_data)
            st.caption(f"Time History (Sensor {measurements[0]['sensorId']})")
            line_chart= alt.Chart(df_sensor).mark_line().encode(
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
        identified, svd_data = perform_analysis(analysis_method, st.session_state.params, st.session_state.measurements)
        st.session_state.identified_modes = identified
        st.session_state.svd_data        = svd_data
        st.session_state.is_analyzing=False

    st.button("Perform Analysis", on_click=run_analysis, disabled=not st.session_state.generation_complete)

    identified_modes = st.session_state.identified_modes
    svd_data = st.session_state.svd_data
    if identified_modes:
        st.markdown(f"### Analysis Results ({analysis_method})")

        # If FDD => show s1, s2, s3
        if analysis_method=="FDD" and svd_data:
            st.markdown("#### Singular Values (SV1, SV2, SV3)")
            df_svd = pd.DataFrame(svd_data)
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

                # If columns missing or empty => skip pinned shape logic
                needed_cols = {"position","trueShape","identifiedShape"}
                if not needed_cols.issubset(df_mode.columns) or df_mode.empty:
                    st.write("No shape data to plot for this mode.")
                    continue

                # Pinned shape approach
                df_true_line = pd.concat([
                    pd.DataFrame({"position":[0],"trueShapeN":[0]}),
                    df_mode[["position","trueShape"]].rename(columns={"trueShape":"trueShapeN"}),
                    pd.DataFrame({"position":[params['bridgeLength']], "trueShapeN":[0]})
                ], ignore_index=True)

                df_ident_line= pd.concat([
                    pd.DataFrame({"position":[0],"identifiedShapeN":[0]}),
                    df_mode[["position","identifiedShape"]].rename(columns={"identifiedShape":"identifiedShapeN"}),
                    pd.DataFrame({"position":[params['bridgeLength']], "identifiedShapeN":[0]})
                ], ignore_index=True)

                line_true = alt.Chart(df_true_line).mark_line(strokeDash=[5,3], color="gray").encode(
                    x=alt.X("position", title="Position (m)"),
                    y=alt.Y("trueShapeN", title="Normalized Amplitude", scale=alt.Scale(domain=[-1.1,1.1]))
                )
                line_ident= alt.Chart(df_ident_line).mark_line(color="red").encode(
                    x="position",
                    y=alt.Y("identifiedShapeN", scale=alt.Scale(domain=[-1.1,1.1]))
                )
                points_ident= alt.Chart(df_mode).mark_point(color="red", filled=True, size=50).encode(
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
    **KeyError** was caused by missing `"trueShape"` or `"identifiedShape"` columns in mode shape data.
    - We now ensure the **SSI** method returns dummy shape data for each mode (so columns exist).
    - We also **check** columns exist before slicing them for pinned shape visualization.

    ### References
    1. Brincker, R., & Ventura, C. (2015). *Introduction to Operational Modal Analysis*. Wiley.
    2. Peeters, B., & De Roeck, G. (2001). *Stochastic System Identification for Operational Modal Analysis: A Review*.
       Journal of Dynamic Systems, Measurement, and Control, 123(4), 659-667.
    3. Rainieri, C., & Fabbrocino, G. (2014). *Operational Modal Analysis of Civil Engineering Structures*. Springer.
    4. Au, S. K. (2017). *Operational Modal Analysis: Modeling, Bayesian Inference, Uncertainty Laws*. Springer.
    5. Brincker, R., Zhang, L., & Andersen, P. (2000). *Modal identification from ambient responses using frequency domain decomposition*.
       In Proceedings of the 18th International Modal Analysis Conference (IMAC), San Antonio, Texas.
    """)


###############################################################################
# 4) If invoked directly, run main()
###############################################################################
if __name__=="__main__":
    main()
