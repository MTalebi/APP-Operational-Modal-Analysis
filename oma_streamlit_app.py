import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from typing import List, Dict

###############################################################################
# 1) Streamlit requires set_page_config() as the very first command.
###############################################################################
st.set_page_config(
    page_title="FDD-Only OMA Tool",
    layout="wide"
)

###############################################################################
# 2) Helper functions that do not call st.* 
###############################################################################

def cpsd_fft_based(x: np.ndarray, y: np.ndarray, Fs: float):
    """
    Optional helper to compute cross-spectral density (CSD) via direct FFT
    instead of scipy.signal.csd, if desired.
    """
    from scipy.fft import fft, fftfreq
    n = len(x)
    X = fft(x, n=n)
    Y = fft(y, n=n)
    Gxy = (X * np.conjugate(Y)) / n
    freq = fftfreq(n, d=1.0 / Fs)
    half = n // 2 + 1
    return freq[:half], Gxy[:half]

def calculate_mac(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the Modal Assurance Criterion (MAC) between two vectors.
    MAC = (|vec1^T vec2|^2) / ((vec1^T vec1)*(vec2^T vec2)).
    """
    numerator = (np.sum(vec1 * vec2))**2
    denominator = np.sum(vec1**2) * np.sum(vec2**2)
    if denominator < 1e-16:
        return 0.0
    return float(numerator / denominator)

###############################################################################
# 3) FDD Implementation
###############################################################################

def FDDAuto(
    Acc: np.ndarray,
    Fs: float,
    Num_Modes_Considered: int,
    EstimeNaturalFreqs: list,
    csd_method: str = 'cpsd'
):
    """
    Frequency Domain Decomposition (FDD):
      1) Builds the Cross-Power Spectral Density for each frequency line.
      2) Performs SVD at each line to get the first three singular values.
      3) Identifies modes by peak-picking near the user's estimated frequencies.
      4) Re-SVD at those peak frequencies to retrieve mode shapes.

    Returns:
      (IdentFreqs, ModeShapes, freq_axis, s1, s2, s3)
        IdentFreqs  : list of identified freq for each mode
        ModeShapes  : shape (#sensors, Num_Modes_Considered)
        freq_axis   : array of frequencies up to Nyquist
        s1, s2, s3  : arrays of singular values at each freq line
    """
    from scipy.signal import csd as csd_func
    from scipy.linalg import svd

    # We assume Acc is shape (num_samples, num_sensors).
    # Build cross-power spectra.
    # For speed, let us do a smaller nperseg to avoid heavy computations:
    # e.g. nperseg= Acc.shape[0]//4 if Acc.shape[0] is large
    n_samples, n_sensors = Acc.shape

    PSD = np.zeros((n_samples // 2 + 1, n_sensors, n_sensors), dtype=complex)
    F   = np.zeros((n_samples // 2 + 1, n_sensors, n_sensors))

    # We'll build the PSD matrix (f x #sensors x #sensors)
    for i in range(n_sensors):
        for j in range(n_sensors):
            if csd_method == 'cpsd':
                f_ij, PSD_ij = csd_func(
                    Acc[:, i],
                    Acc[:, j],
                    fs=Fs,
                    nperseg=max(256, n_samples//4),   # reduce for speed
                    noverlap=None,
                    nfft=n_samples
                )
            else:
                f_ij, PSD_ij = cpsd_fft_based(Acc[:, i], Acc[:, j], Fs)

            F[:, i, j]   = f_ij
            PSD[:, i, j] = PSD_ij

    # We'll gather them in lists
    PSD_list = []
    Freq_list= []
    for k in range(PSD.shape[0]):
        PSD_list.append(PSD[k, :, :])
        Freq_list.append(F[k, :, :])

    # We'll do SVD at each freq line => store s1, s2, s3
    n_lines = len(PSD_list)
    s1_arr = np.zeros(n_lines)
    s2_arr = np.zeros(n_lines)
    s3_arr = np.zeros(n_lines)

    # SVD each freq
    from scipy.linalg import svd
    for line_idx in range(n_lines):
        mat = PSD_list[line_idx]
        U, S, Vh = svd(mat, full_matrices=False)
        if len(S) > 0: s1_arr[line_idx] = S[0]
        if len(S) > 1: s2_arr[line_idx] = S[1]
        if len(S) > 2: s3_arr[line_idx] = S[2]

    freq_axis = np.stack(Freq_list)[:, 1, 1].flatten()

    # Peak pick near each estimated freq
    IdentFreqs = []
    bandwidth  = 0.3
    for freq_est in EstimeNaturalFreqs:
        lowb = max(0, freq_est - bandwidth)
        highb= freq_est + bandwidth
        subset_idx = np.where((freq_axis>=lowb)&(freq_axis<=highb))[0]
        if len(subset_idx)==0:
            # fallback => nearest freq
            best_idx = np.argmin(np.abs(freq_axis - freq_est))
        else:
            # pick max of s1 in that range
            best_idx = subset_idx[np.argmax(s1_arr[subset_idx])]
        IdentFreqs.append(freq_axis[best_idx])

    # Re-SVD => shapes
    Num_Modes_Considered= min(Num_Modes_Considered, len(IdentFreqs))
    ModeShapes= np.zeros((n_sensors, Num_Modes_Considered))
    for mode_idx in range(Num_Modes_Considered):
        # freq is IdentFreqs[mode_idx]
        # find that freq in freq_axis
        pick_freq= IdentFreqs[mode_idx]
        # let's locate the nearest line
        line_index= np.argmin(np.abs(freq_axis - pick_freq))
        mat_line= PSD_list[line_index]
        U_line, S_line, Vh_line = svd(mat_line)
        # first singular vector
        phi_cpx= U_line[:, 0]
        # Align phase
        Y= np.imag(phi_cpx)
        X= np.column_stack((np.real(phi_cpx), np.ones(len(phi_cpx))))
        A_ = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
        theta = np.arctan(A_[0])
        phi_rot= phi_cpx* np.exp(-1j*theta)
        phi_real= np.real(phi_rot)
        # normalize
        if np.max(np.abs(phi_real))<1e-12:
            phi_real= np.zeros(len(phi_real))
        else:
            phi_real/= np.max(np.abs(phi_real))
        ModeShapes[:, mode_idx]= phi_real

    return IdentFreqs, ModeShapes, freq_axis, s1_arr, s2_arr, s3_arr

def finalize_fdd_mode(
    mode_number: int,
    freq_true: float,
    freq_ident: float,
    shape_ident: np.ndarray,
    measurements: List[Dict],
    bridge_length: float
):
    """
    Builds a dict describing one identified mode from FDD, 
    including pinned-shape data for each sensor.
    """
    import pandas as pd
    sensor_data=[]
    for s_idx, sens in enumerate(measurements):
        x_sens= sens["position"]
        # ground-truth shape => sin(mode_number*pi*x / L)
        val_true= np.sin(mode_number*np.pi*x_sens/ bridge_length)
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

    max_t= df["trueShape"].abs().max()
    max_i= df["identifiedShape"].abs().max()
    if max_t<1e-12: max_t=1
    if max_i<1e-12: max_i=1
    df["trueShapeN"]= df["trueShape"]/max_t
    df["identifiedShapeN"]= df["identifiedShape"]/max_i
    freq_err= 0.0
    if freq_true!=0:
        freq_err= ((freq_ident- freq_true)/ freq_true)*100.0
    mac_val= calculate_mac(df["trueShapeN"], df["identifiedShapeN"])

    mode_shapes=[]
    for row in df.itertuples():
        mode_shapes.append({
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
        "modeShapes": mode_shapes
    }

###############################################################################
# 4) Synthetic Data Generation
###############################################################################

def generate_synthetic_data(params: dict):
    """
    Creates synthetic signals for a 3-mode structure + noise.
    Returns a list of sensor dicts => { "sensorId", "position", "data": [...] }.
    """
    import numpy as np

    L= params["bridgeLength"]
    n_sensors= params["numSensors"]
    mo1= params["mode1"]
    mo2= params["mode2"]
    mo3= params["mode3"]
    noise= params["noiseLevel"]
    fs=   params["samplingFreq"]
    dur=  params["duration"]

    dt= 1.0/fs
    N= int(np.floor(dur*fs))
    sensor_positions= np.linspace(
        L/(n_sensors+1),
        L*(n_sensors/(n_sensors+1)),
        n_sensors
    )

    time_vec= np.arange(N)* dt
    resp1= mo1["amp"]* np.sin(2*np.pi* mo1["freq"]* time_vec)
    resp2= mo2["amp"]* np.sin(2*np.pi* mo2["freq"]* time_vec)
    resp3= mo3["amp"]* np.sin(2*np.pi* mo3["freq"]* time_vec)

    shape_list=[]
    for x in sensor_positions:
        s1= np.sin(np.pi*x/L)
        s2= np.sin(2*np.pi*x/L)
        s3= np.sin(3*np.pi*x/L)
        shape_list.append((s1, s2, s3))

    measurements=[]
    for i,x in enumerate(sensor_positions):
        shape1, shape2, shape3= shape_list[i]
        data_list=[]
        for t in range(N):
            M1= shape1* resp1[t]
            M2= shape2* resp2[t]
            M3= shape3* resp3[t]
            noise_val= noise*(2*np.random.rand()-1)
            total_acc= M1+ M2+ M3+ noise_val
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

###############################################################################
# 5) Perform FDD Analysis
###############################################################################

def perform_fdd(params: dict, measurements: List[Dict]):
    """
    Conducts the entire FDD procedure:
      - Creates an Acc matrix (samples x sensors)
      - Calls FDDAuto
      - Builds an identified_modes list with pinned-shape data
      - Returns (identified_modes, svd_data)
    """
    import numpy as np

    # Build acceleration matrix => shape (N x #sensors)
    n_sensors= len(measurements)
    n_samples= len(measurements[0]["data"])
    Acc= np.zeros((n_samples, n_sensors))
    for i, sens in enumerate(measurements):
        Acc[:, i]= [pt["acceleration"] for pt in sens["data"]]

    # We'll pick the three modes
    f1= params["mode1"]["freq"]
    f2= params["mode2"]["freq"]
    f3= params["mode3"]["freq"]
    freq_est= [f1, f2, f3]

    # Call FDDAuto
    IdentFreqs, ModeShapes, freq_axis, s1, s2, s3= FDDAuto(
        Acc, params["samplingFreq"],
        Num_Modes_Considered= 3,
        EstimeNaturalFreqs= freq_est,
        csd_method='cpsd'
    )

    # Build the identified modes array
    identified_modes=[]
    L= params["bridgeLength"]
    true_frqs= [f1, f2, f3]
    for i, mode_num in enumerate([1,2,3]):
        freq_true= true_frqs[i]
        freq_ident= IdentFreqs[i]
        shape_ident= ModeShapes[:, i]
        one_mode= finalize_fdd_mode(mode_num, freq_true, freq_ident, shape_ident, measurements, L)
        identified_modes.append(one_mode)

    # Create the "svd_data" for external plotting of s1, s2, s3
    svd_data=[]
    for i in range(len(freq_axis)):
        svd_data.append({
            "frequency": freq_axis[i],
            "sv1": s1[i],
            "sv2": s2[i],
            "sv3": s3[i]
        })
    return identified_modes, svd_data

###############################################################################
# 6) Streamlit "main" function
###############################################################################

def main():
    st.markdown("""
    # Frequency Domain Decomposition (FDD) Tool
    This app demonstrates how to:
    - Generate synthetic vibration data for a structure with three modes + random noise.
    - Compute Cross-Power Spectra and perform **FDD**:
       1. Do an SVD at each frequency line to get the first three singular values.
       2. Identify the modes by “peak picking” near user-estimated frequencies.
       3. Retrieve the mode shapes and compare with the theoretical sine-based shapes 
          at each sensor's position.

    **Steps**:
    1. Adjust the simulation parameters.
    2. Click "Generate Synthetic Data".
    3. Click "Perform FDD Analysis".
    4. Observe the singular-value plots (SV1, SV2, SV3) and the pinned-shape results.
    """)

    with st.expander("Educational Notes", expanded=False):
        st.write("""
        **FDD** relies on cross-power spectral density. Each frequency line 
        yields a matrix whose largest singular value typically shows resonant peaks. 
        By focusing on these peaks, we identify the modes. Then, the corresponding 
        singular vectors give the shapes.  
        
        **Controls**:
        - "bridgeLength", "numSensors" help define the geometry.
        - "mode1", "mode2", "mode3" specify the structure's frequency content and amplitude.
        - "noiseLevel" sets the random noise amplitude.
        - "samplingFreq" and "duration" define how many samples we collect.

        After analyzing, we compare the pinned shape: the "true" shape is a sine function 
        of the mode number, while the "identified" shape comes from the SVD at the resonant line.
        """)

    # Session-state
    if "params" not in st.session_state:
        st.session_state.params= {
            "bridgeLength": 30.0,
            "numSensors":   10,
            "mode1": {"freq": 2.0, "amp": 50.0},
            "mode2": {"freq": 8.0, "amp": 10.0},
            "mode3": {"freq": 18.0, "amp": 1.0},
            "noiseLevel":   2.0,
            "samplingFreq": 100.0,
            "duration":     30.0
        }
    if "measurements" not in st.session_state:
        st.session_state.measurements= []
    if "identified_modes" not in st.session_state:
        st.session_state.identified_modes= []
    if "svd_data" not in st.session_state:
        st.session_state.svd_data= []
    if "generation_complete" not in st.session_state:
        st.session_state.generation_complete= False
    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing= False

    params= st.session_state.params

    st.header("Simulation Setup")
    c1, c2= st.columns(2)
    with c1:
        params["numSensors"]= st.slider("Number of Sensors",5,20,int(params["numSensors"]),1)
    with c2:
        params["bridgeLength"]= st.slider("Bridge Length (m)",10.0,100.0,float(params["bridgeLength"]),5.0)

    col1, col2= st.columns(2)
    with col1:
        params["mode1"]["freq"]= st.slider("Mode 1 Frequency (Hz)",0.5,5.0,params["mode1"]["freq"],0.1)
        params["mode1"]["amp"] = st.slider("Mode 1 Amplitude",10.0,100.0,params["mode1"]["amp"],5.0)

        params["mode2"]["freq"]= st.slider("Mode 2 Frequency (Hz)",5.0,15.0,params["mode2"]["freq"],0.5)
        params["mode2"]["amp"] = st.slider("Mode 2 Amplitude",1.0,50.0,params["mode2"]["amp"],1.0)
    with col2:
        params["mode3"]["freq"]= st.slider("Mode 3 Frequency (Hz)",15.0,25.0,params["mode3"]["freq"],0.5)
        params["mode3"]["amp"] = st.slider("Mode 3 Amplitude",0.1,10.0,params["mode3"]["amp"],0.1)

    with st.expander("Advanced Parameters", expanded=False):
        adv1, adv2= st.columns(2)
        with adv1:
            params["noiseLevel"]= st.slider("Noise Level",0.0,10.0,params["noiseLevel"],0.5)
        with adv2:
            params["samplingFreq"]= st.slider("Sampling Frequency (Hz)",20.0,200.0,params["samplingFreq"],10.0)

    st.session_state.params= params

    st.markdown("---")
    st.subheader("Data Generation")
    if st.button("Generate Synthetic Data"):
        st.session_state.measurements= generate_synthetic_data(st.session_state.params)
        st.session_state.identified_modes=[]
        st.session_state.svd_data=[]
        st.session_state.generation_complete= True

    measurements= st.session_state.measurements
    if st.session_state.generation_complete and measurements:
        st.markdown("**Preview: Synthetic Measurements**")
        # show partial time-history from sensor #1
        first_data= measurements[0]["data"][:500]
        if first_data:
            df_sensor= pd.DataFrame(first_data)
            st.caption(f"Acceleration Time History (Sensor {measurements[0]['sensorId']})")
            line_chart= alt.Chart(df_sensor).mark_line().encode(
                x=alt.X("time", title="Time (s)"),
                y=alt.Y("acceleration", title="Acceleration"),
                tooltip=["time","acceleration"]
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

        # sensor positions
        st.caption("**Sensor Positions**")
        df_pos= pd.DataFrame({
            "sensorId":[m["sensorId"] for m in measurements],
            "position":[m["position"] for m in measurements],
            "y":[0.5]*len(measurements)
        })
        scatter_chart= alt.Chart(df_pos).mark_point(size=100, shape="triangle").encode(
            x=alt.X("position",title="Position (m)"),
            y=alt.Y("y",title="",scale=alt.Scale(domain=[0,1])),
            tooltip=["sensorId","position"]
        ).properties(height=150)
        st.altair_chart(scatter_chart,use_container_width=True)

        st.caption(" | ".join([f"{m['sensorId']}: {m['position']:.2f} m" for m in measurements]))

    st.markdown("---")
    st.subheader("Perform FDD Analysis")

    def run_fdd():
        st.session_state.is_analyzing= True
        identified, sv_data= perform_fdd(st.session_state.params, st.session_state.measurements)
        st.session_state.identified_modes= identified
        st.session_state.svd_data= sv_data
        st.session_state.is_analyzing= False

    st.button("Perform FDD Analysis", on_click=run_fdd, disabled= not st.session_state.generation_complete)

    if st.session_state.identified_modes:
        st.markdown("### Analysis Results (FDD)")

        # singular values
        svd_data= st.session_state.svd_data
        if svd_data:
            st.markdown("#### Singular Values (SV1, SV2, SV3)")
            df_svd= pd.DataFrame(svd_data)
            max_x= 2.0 * st.session_state.params["mode3"]["freq"]

            col_sv= st.columns(3)
            with col_sv[0]:
                chart_sv1= alt.Chart(df_svd).mark_line().encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                    y=alt.Y("sv1", title="Amplitude"),
                    tooltip=["frequency","sv1"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv1, use_container_width=True)

            with col_sv[1]:
                chart_sv2= alt.Chart(df_svd).mark_line(color="red").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                    y=alt.Y("sv2", title="Amplitude"),
                    tooltip=["frequency","sv2"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv2, use_container_width=True)

            with col_sv[2]:
                chart_sv3= alt.Chart(df_svd).mark_line(color="green").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0, max_x])),
                    y=alt.Y("sv3", title="Amplitude"),
                    tooltip=["frequency","sv3"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv3, use_container_width=True)

        # identified modes
        identified_modes= st.session_state.identified_modes
        st.markdown("#### Identified Modal Parameters")
        df_modes= pd.DataFrame([
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
        shape_cols= st.columns(len(identified_modes))
        for i, mode_info in enumerate(identified_modes):
            with shape_cols[i]:
                st.markdown(f"**Mode {mode_info['modeNumber']}**")
                df_mode= pd.DataFrame(mode_info["modeShapes"])

                needed_cols= {"position","trueShape","identifiedShape"}
                if not needed_cols.issubset(df_mode.columns) or df_mode.empty:
                    st.write("No shape data available for pinned-shape plot.")
                    continue

                # pinned shape approach
                df_true_line= pd.concat([
                    pd.DataFrame({"position":[0],"trueShapeN":[0]}),
                    df_mode[["position","trueShape"]].rename(columns={"trueShape":"trueShapeN"}),
                    pd.DataFrame({"position":[st.session_state.params['bridgeLength']], "trueShapeN":[0]})
                ], ignore_index=True)

                df_ident_line= pd.concat([
                    pd.DataFrame({"position":[0],"identifiedShapeN":[0]}),
                    df_mode[["position","identifiedShape"]].rename(columns={"identifiedShape":"identifiedShapeN"}),
                    pd.DataFrame({"position":[st.session_state.params['bridgeLength']], "identifiedShapeN":[0]})
                ], ignore_index=True)

                line_true= alt.Chart(df_true_line).mark_line(strokeDash=[5,3], color="gray").encode(
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
                chart_mode= (line_true + line_ident + points_ident).properties(height=250).interactive()
                st.altair_chart(chart_mode, use_container_width=True)

                st.caption(
                    f"**Freq:** {float(mode_info['identifiedFrequency']):.2f} Hz "
                    f"(True: {float(mode_info['trueFrequency']):.2f} Hz), "
                    f"**MAC:** {mode_info['mac']}, "
                    f"**Error:** {mode_info['frequencyError']}%"
                )

    st.markdown("---")
    st.subheader("References & Additional Info")
    st.markdown("""
    **Key Points**:
    - FDD looks at singular values of the cross-power spectra. Peaks in these singular values 
      indicate possible resonant modes.
    - We “peak pick” near your estimated frequencies. The corresponding singular vectors 
      define the identified mode shapes.
    - We compare each identified shape to a simple sine-based “true shape” for pinned-pinned 
      structural assumptions, computing the MAC for shape consistency.

    **References**:
    1. Brincker, R., & Ventura, C. (2015). *Introduction to Operational Modal Analysis*. Wiley.
    2. Brincker, R., Zhang, L., & Andersen, P. (2000). *Modal identification from ambient responses 
       using frequency domain decomposition.* Proceedings of the 18th International Modal Analysis Conference (IMAC).
    3. Rainieri, C., & Fabbrocino, G. (2014). *Operational Modal Analysis of Civil Engineering Structures*. Springer.

    Thank you for exploring this FDD-based OMA tool.
    """)

###############################################################################
# 7) If invoked directly, run main()
###############################################################################
if __name__=="__main__":
    main()
