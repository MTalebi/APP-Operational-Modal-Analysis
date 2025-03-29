import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from scipy.signal import csd as cpsd
from numpy.linalg import svd, inv, pinv, eig
from scipy.fft import fft, fftfreq
###############################################################################
# (1) Streamlit must call set_page_config() before other UI commands.
###############################################################################
st.set_page_config(
    page_title="OMA Tool",
    layout="wide"
)

###############################################################################
# (2) General helper functions and classes (no Streamlit calls here).
###############################################################################

def cpsd_fft_based(x, y, Fs):
    """
    Optional helper for cross-spectrum via direct FFT 
    instead of scipy.signal.csd.
    """
    n = len(x)
    X = fft(x, n=n)
    Y = fft(y, n=n)
    Gxy = (X * np.conjugate(Y)) / n
    freq = fftfreq(n, d=1.0 / Fs)
    half = n // 2 + 1
    return freq[:half], Gxy[:half]

def calculate_mac(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Modal Assurance Criterion (MAC):
      MAC = |v1^T v2|^2 / ((v1^T v1) * (v2^T v2))
    Values near 1.0 imply strong similarity.
    """
    numerator = (np.sum(v1 * v2))**2
    denominator = np.sum(v1**2)*np.sum(v2**2)
    if denominator < 1e-16:
        return 0.0
    return float(numerator / denominator)

###############################################################################
# (3) FDD Approach
###############################################################################

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
    Frequency Domain Decomposition:
      1) Builds cross-power spectral matrix from data.
      2) SVD at each frequency => singular values.
      3) Picks peaks near EstimeNaturalFreqs => identified modes.
      4) Returns freq axis + s1,s2,s3 and shapes for external plotting.

    Acc is assumed shape: (N x channels), 
    where N = #samples, channels = #sensors.
    """

    PSD = np.zeros((Acc.shape[0]//2 + 1, Acc.shape[1], Acc.shape[1]), dtype=complex)
    F   = np.zeros((Acc.shape[0]//2 + 1, Acc.shape[1], Acc.shape[1]))

    # Build cross-power spectra
    for I in range(Acc.shape[1]):
        for J in range(Acc.shape[1]):
            if csd_method=='cpsd':
                f_ij, PSD_ij = cpsd(
                    Acc[:, I], Acc[:, J],
                    nperseg=Acc.shape[0]//3,
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

    # Identify modes from PSD_list
    Frq, Phi_Real, freq_axis, s1, s2, s3 = _identifier_fdd(
        PSD_list, F_list, Num_Modes_Considered, EstimeNaturalFreqs, set_singular_ylim, plot_internally
    )
    return Frq, Phi_Real, freq_axis, s1, s2, s3

def _identifier_fdd(PSD_list, F_list, Num_Modes_Considered,
                    EstimeNaturalFreqs, set_singular_ylim, plot_internally):
    """
    Internal subroutine for FDD:
     - SVD at each freq line => s1,s2,s3
     - Peak-pick near EstimeNaturalFreqs => identified freq
     - Re-SVD at each peak => shapes
    """


    n_freqs = len(PSD_list)
    n_chan  = PSD_list[0].shape[0]
    s1_arr  = np.zeros(n_freqs)
    s2_arr  = np.zeros(n_freqs)
    s3_arr  = np.zeros(n_freqs)

    # (a) SVD each freq
    for i in range(n_freqs):
        mat = PSD_list[i]
        u, s, vh= svd(mat)
        if len(s)>0: s1_arr[i]= s[0]
        if len(s)>1: s2_arr[i]= s[1]
        if len(s)>2: s3_arr[i]= s[2]

    freq_arr= np.stack(F_list)[:,1,1].flatten()

    # (b) Peak pick near each EstimeNaturalFreq
    bandwidth= 0.3
    Fp=[]
    for k in range(Num_Modes_Considered):
        f_est= EstimeNaturalFreqs[k]
        lowb= max(0, f_est-bandwidth)
        highb= f_est+ bandwidth
        subset_idx= np.where((freq_arr>= lowb)&(freq_arr<= highb))[0]
        if len(subset_idx)==0:
            best_idx= np.argmin(np.abs(freq_arr- f_est))
        else:
            best_idx= subset_idx[np.argmax(s1_arr[subset_idx])]
        Fp.append([best_idx, freq_arr[best_idx]])
    Fp= np.array(Fp)
    sr= np.argsort(Fp[:,1])
    Fp= Fp[sr,:]
    identified_freqs= Fp[:,1]

    # (c) Re-SVD at each peak => shapes
    Phi_Real= np.zeros((n_chan, Num_Modes_Considered))
    for j in range(Num_Modes_Considered):
        idx_peak= int(Fp[j,0])
        mat= PSD_list[idx_peak]
        u, s, vh= svd(mat)
        phi_cpx= u[:,0]
        # align phase
        Y= np.imag(phi_cpx)
        X= np.column_stack((np.real(phi_cpx), np.ones(len(phi_cpx))))
        A= np.linalg.pinv(X.T@X)@(X.T@Y)
        theta= np.arctan(A[0])
        phi_rot= phi_cpx*np.exp(-1j*theta)
        phi_rot_real= np.real(phi_rot)
        phi_rot_real/= np.max(np.abs(phi_rot_real))
        Phi_Real[:, j]= phi_rot_real

    return identified_freqs, Phi_Real, freq_arr, s1_arr, s2_arr, s3_arr

def finalize_mode_entry(mode_number, freq_true, freq_ident,
                        shape_ident, measurements, L):
    """
    Creates a dictionary describing one mode:
      - modeNumber, trueFrequency, identifiedFrequency
      - frequencyError, mac
      - modeShapes => list of dicts with position, sensorId, trueShape, identifiedShape
    """
    sensor_data=[]
    for i, sensor in enumerate(measurements):
        x_sens= sensor["position"]
        val_true= np.sin(mode_number*np.pi*x_sens/L)
        val_ident= shape_ident[i]
        sensor_data.append({
            "sensorId": sensor["sensorId"],
            "position": x_sens,
            "trueShape": val_true,
            "identifiedShape": val_ident
        })
    df= pd.DataFrame(sensor_data)
    if df.empty:
        return {
            "modeNumber": mode_number,
            "trueFrequency": freq_true,
            "identifiedFrequency": freq_ident,
            "frequencyError":"N/A",
            "mac":"N/A",
            "modeShapes":[]
        }
    # normalize => MAC
    max_t= df["trueShape"].abs().max()
    max_i= df["identifiedShape"].abs().max()
    if max_t<1e-12: max_t=1.0
    if max_i<1e-12: max_i=1.0
    df["trueShapeN"]= df["trueShape"]/ max_t
    df["identifiedShapeN"]= df["identifiedShape"]/ max_i
    signs = np.sign(df["trueShape"].max() * df.loc[df["trueShape"].idxmax(), "identifiedShape"])
    df["identifiedShapeN"]= df["identifiedShape"] * signs

    mac_val= calculate_mac(df["trueShapeN"].values, df["identifiedShapeN"].values)
    freq_err=0.0
    if freq_true!=0:
        freq_err= ((freq_ident- freq_true)/ freq_true)*100.0

    # build final shape array
    shape_list=[]
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

def perform_fdd(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    1) Assembles data into (samples x sensors) format
    2) Calls FDDAuto for cross-spectral SVD
    3) Identifies 3 modes => finalize, compute MAC, shapes
    4) Also returns s1,s2,s3 => we can plot them externally in Altair
    """
    n_sensors= len(measurements)
    n_samples= len(measurements[0]["data"])
    data_mat= np.zeros((n_samples, n_sensors))
    for i, sensor in enumerate(measurements):
        data_mat[:, i]= [pt["acceleration"] for pt in sensor["data"]]

    # approximate frequencies for peak picking
    f1= params["mode1"]["freq"]
    f2= params["mode2"]["freq"]
    f3= params["mode3"]["freq"]
    est_frqs= [f1,f2,f3]
    Frq, Phi_Real, freq_axis, s1, s2, s3= FDDAuto(
        Acc= data_mat,
        Fs= params["samplingFreq"],
        Num_Modes_Considered= 3,
        EstimeNaturalFreqs= est_frqs,
        set_singular_ylim= None,
        csd_method='cpsd',
        plot_internally= False
    )

    identified=[]
    L= params["bridgeLength"]
    t_frqs= [f1,f2,f3]
    for i, mode_num in enumerate([1,2,3]):
        freq_true= t_frqs[i]
        freq_ident= Frq[i]
        shape_ident= Phi_Real[:, i]
        identified_mode= finalize_mode_entry(mode_num, freq_true, freq_ident,
                                            shape_ident, measurements, L)
        identified.append(identified_mode)

    svd_data=[]
    for i in range(len(freq_axis)):
        svd_data.append({
            "frequency": freq_axis[i],
            "sv1": s1[i],
            "sv2": s2[i],
            "sv3": s3[i]
        })
    return identified, svd_data

###############################################################################
# 4) Real SSI Implementation (no made-up shapes)
###############################################################################

def real_ssi_method(output, fs, ncols, nrows, cut):
    """
    Translates your MATLAB-based SSI snippet to Python, returning 
    frequencies, damping, and real-valued mode shapes (no random shapes).

    Output shape => (#outputs, #samples). 
    If that is reversed, we transpose. 
    """

    outs, npts= output.shape
    if outs> npts:
        output= output.T
        outs, npts= output.shape

    # clamp ncols, nrows to safe values
    ncols= min(ncols, npts)
    nrows= min(nrows, outs*(npts//2))

    brows= nrows// outs
    nrows= outs*brows
    bcols= ncols//1
    ncols= 1*bcols
    m=1
    q= outs

    # Build Hankel Yp, Yf
    Yp= np.zeros((nrows//2, ncols), dtype=float)
    Yf= np.zeros((nrows//2, ncols), dtype=float)
    half_brows= brows//2

    for ii in range(1, half_brows+1):
        for jj in range(1, bcols+1):
            r_start= (ii-1)*q
            r_end=   ii*q
            c_start= (jj-1)*m
            c_end=   jj*m
            left= ((jj-1)+(ii-1))*m
            right=((jj)+(ii-1))*m
            if right> npts:
                break
            Yp[r_start:r_end, c_start:c_end]= output[:, left:right]

    for ii in range(half_brows+1, brows+1):
        i_= ii- half_brows
        for jj in range(1, bcols+1):
            r_start= (i_-1)*q
            r_end=   i_*q
            c_start= (jj-1)*m
            c_end=   jj*m
            left= ((jj-1)+(ii-1))*m
            right=((jj)+(ii-1))*m
            if right> npts:
                break
            Yf[r_start:r_end, c_start:c_end]= output[:, left:right]

    # Projection
    O= Yf @ Yp.T @ pinv(Yp @ Yp.T) @ Yp

    # SVD
    R1, Sigma1, S1= svd(O, full_matrices=False)
    sv= Sigma1
    D= np.diag(np.sqrt(sv[:cut]))
    Rn= R1[:, :cut]
    Sn= S1[:cut, :]

    Obs= Rn @ D
    top_rows= (nrows//2)- q
    Obs_up=   Obs[:top_rows, :]
    Obs_down= Obs[q:nrows//2, :]
    A= pinv(Obs_up)@ Obs_down
    C= Obs[:q, :]

    # eigen => freq, damping
    lam, V= eig(A)
    s= np.log(lam)* fs
    zeta= -np.real(s)/ np.abs(s)*100.0
    fd=   np.imag(s)/(2.0*np.pi)
    shapes= C@ V

    # Additional measures, e.g. partfac:
    InvV= inv(V)
    partfac= np.std( (InvV @ D @ Sn).T, axis=0)

    # placeholders for EMAC, MPC, CMI
    EMAC= np.ones(cut)* 80
    MPC=  np.ones(cut)* 85
    CMI=  EMAC*MPC/100.0

    # sort ascending freq
    idx= np.argsort(fd)
    fd= fd[idx]
    zeta= zeta[idx]
    shapes= shapes[:, idx]
    partfac= partfac[idx]
    EMAC= EMAC[idx]
    MPC=  MPC[idx]
    CMI=  CMI[idx]

    # remove freq <0 or freq> fs/2
    mask= (fd> 0)&(fd< 0.499*fs)
    fd= fd[mask]
    zeta= zeta[mask]
    shapes= shapes[:, mask]
    partfac= partfac[mask]
    EMAC= EMAC[mask]
    MPC=  MPC[mask]
    CMI=  CMI[mask]

    # convert damped => undamped freq => fd1= fd/sqrt(1-(zeta/100)^2)
    fd1= fd/ np.sqrt(np.maximum(1-(zeta/100.0)**2, 1e-16))
    idx2= np.argsort(fd1)
    fd1= fd1[idx2]
    zeta= zeta[idx2]
    shapes= shapes[:, idx2]
    partfac= partfac[idx2]
    EMAC= EMAC[idx2]
    MPC=  MPC[idx2]
    CMI=  CMI[idx2]

    # Force shapes real => float
    nmodes= shapes.shape[1]
    shapes_real= np.zeros_like(shapes, dtype=float)
    for jj in range(nmodes):
        abscol= np.abs(shapes[:, jj])
        rowmax= np.argmax(abscol)
        b= -np.angle(shapes[rowmax, jj])
        shape_rot= shapes[:, jj]* np.exp(1j*b)
        shape_rot= np.real(shape_rot)
        denom= np.sqrt(np.sum(shape_rot**2))
        if denom<1e-12: denom=1
        shapes_real[:, jj]= (shape_rot/ denom).astype(float)

    shapes= shapes_real

    Result= {
        "Parameters":{
            "NaFreq":    fd1,
            "DampRatio": zeta,
            "ModeShape": shapes
        },
        "Indicators":{
            "EMAC": EMAC,
            "MPC" : MPC,
            "CMI" : CMI,
            "partfac": partfac
        },
        "Matrices":{
            "A": A,
            "C": C
        }
    }
    return Result

def perform_ssi_real(params: dict, measurements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    1) Build data array from measurements => (#outputs, #samples).
    2) run real_ssi_method => retrieve freq, shapes
    3) finalize the first 3 modes => pinned shape
    4) return (identified_modes, []), since we have no s1,s2,s3 for SSI.
    """

    n_sensors= len(measurements)
    n_samples= len(measurements[0]["data"])
    output= np.zeros((n_sensors, n_samples))
    for i, sensor in enumerate(measurements):
        output[i,:]= [pt["acceleration"] for pt in sensor["data"]]

    fs= params["samplingFreq"]
    # heuristics
    nrows= 60
    ncols= max(30, n_samples//2)
    cut= 8

    # run subspace
    result= real_ssi_method(output, fs, ncols, nrows, cut)
    freqs=  result["Parameters"]["NaFreq"]
    shapes= result["Parameters"]["ModeShape"]  # shape => (#sensors, #modes)
    n_found= shapes.shape[1]

    # pick up to 3 modes
    true_freqs= [
        params["mode1"]["freq"],
        params["mode2"]["freq"],
        params["mode3"]["freq"]
    ]
    identified=[]
    L= params["bridgeLength"]
    n_id= min(3, n_found)
    for i in range(n_id):
        freq_ident= freqs[i]
        freq_true= 0.0
        if i<3:
            freq_true= true_freqs[i]
        shape_ident= shapes[:, i]
        # finalize
        mode_entry= finalize_mode_entry(i+1, freq_true, freq_ident,
                                        shape_ident, measurements, L)
        identified.append(mode_entry)

    # SSI doesn't produce s1,s2,s3 => return empty
    return identified, []


###############################################################################
# (5) Data Generation, Analysis Dispatch
###############################################################################

def generate_synthetic_data(params: dict):
    """
    Builds a 3-mode + noise synthetic dataset for demonstration.
    """
    bL= params["bridgeLength"]
    n_sensors= params["numSensors"]
    mo1= params["mode1"]
    mo2= params["mode2"]
    mo3= params["mode3"]
    noise= params["noiseLevel"]
    fs=   params["samplingFreq"]
    dur=  params["duration"]

    dt= 1.0/fs
    N= int(np.floor(dur*fs))
    positions= np.linspace(bL/(n_sensors+1),
                           bL*(n_sensors/(n_sensors+1)),
                           n_sensors)
    time_vec= np.arange(N)* dt
    resp1= mo1["amp"]* np.sin(2*np.pi* mo1["freq"]* time_vec)
    resp2= mo2["amp"]* np.sin(2*np.pi* mo2["freq"]* time_vec)
    resp3= mo3["amp"]* np.sin(2*np.pi* mo3["freq"]* time_vec)

    shape_list=[]
    for x in positions:
        s1= np.sin(np.pi*x/bL)
        s2= np.sin(2*np.pi*x/bL)
        s3= np.sin(3*np.pi*x/bL)
        shape_list.append((s1,s2,s3))

    measurements=[]
    for i, x in enumerate(positions):
        shape1, shape2, shape3= shape_list[i]
        data_list=[]
        for t in range(N):
            M1= shape1* resp1[t]
            M2= shape2* resp2[t]
            M3= shape3* resp3[t]
            noise_val= noise*(2*np.random.rand()-1)
            total= M1+M2+M3+ noise_val
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

def perform_analysis(analysis_method: str, params: dict, measurements: List[Dict]):
    """
    Switch between FDD or real SSI, returning 
     (identified_modes, svd_data).
    """
    if analysis_method=="FDD":
        return perform_fdd(params, measurements)
    elif analysis_method=="SSI":
        return perform_ssi_real(params, measurements)
    else:
        return [], []

###############################################################################
# (6) main() with Streamlit UI
###############################################################################
def main():
    st.markdown("""
    # Operational Modal Analysis Tool

    **Developed by**: Mohammad Talebi-Kalaleh  
    Contact: [talebika@ualberta.ca](mailto:talebika@ualberta.ca)
    """)

    st.markdown("""
    This app demonstrates:
    - **Frequency Domain Decomposition (FDD)**: obtains cross-spectral singular values, 
      identifies mode shapes/frequencies from spectral peaks.
    - **SSI**: uses subspace identification to extract a state-space model, 
      then retrieves frequencies, damping, and mode shapes.

    **Instructions**:
    1. Set parameters (bridge length, # of sensors, mode frequencies, etc.).
    2. Click **Generate Synthetic Measurements**.
    3. Pick an **Analysis Method** (FDD or SSI).
    4. Press **Perform Analysis** to see the results.
    """)


    # session-state
    if "params" not in st.session_state:
        st.session_state.params= {
            "bridgeLength":30.0,
            "numSensors":10,
            "mode1":{"freq":2.0,"amp":50.0},
            "mode2":{"freq":8.0,"amp":10.0},
            "mode3":{"freq":18.0,"amp":1.0},
            "noiseLevel":2.0,
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

    params= st.session_state.params

    st.header("Set Simulation Parameters")

    c1, c2= st.columns(2)
    with c1:
        params["numSensors"]= st.slider("Number of Sensors",5,20,int(params["numSensors"]),1)
    with c2:
        params["bridgeLength"]= st.slider("Bridge Length (m)",10.0,100.0,float(params["bridgeLength"]),5.0)

    col1, col2= st.columns(2)
    with col1:
        params["mode1"]["freq"] = st.slider("Mode 1 Frequency (Hz)",0.5,5.0,params["mode1"]["freq"],0.1)
        params["mode1"]["amp"]  = st.slider("Mode 1 Amplitude",10.0,100.0,params["mode1"]["amp"],5.0)
        params["mode2"]["freq"] = st.slider("Mode 2 Frequency (Hz)",5.0,15.0,params["mode2"]["freq"],0.5)
        params["mode2"]["amp"]  = st.slider("Mode 2 Amplitude",1.0,50.0,params["mode2"]["amp"],1.0)
    with col2:
        params["mode3"]["freq"] = st.slider("Mode 3 Frequency (Hz)",15.0,25.0,params["mode3"]["freq"],0.5)
        params["mode3"]["amp"]  = st.slider("Mode 3 Amplitude",0.1,10.0,params["mode3"]["amp"],0.1)

    with st.expander("Advanced Parameters", expanded=False):
        adv1, adv2= st.columns(2)
        with adv1:
            params["noiseLevel"]= st.slider("Noise Level",0.0,10.0,params["noiseLevel"],0.5)
        with adv2:
            params["samplingFreq"]= st.slider("Sampling Frequency (Hz)",20.0,200.0,params["samplingFreq"],10.0)

    st.session_state.params= params

    st.markdown("---")
    st.subheader("Generate Data")
    if st.button("Generate Synthetic Measurements"):
        st.session_state.measurements= generate_synthetic_data(st.session_state.params)
        st.session_state.identified_modes=[]
        st.session_state.svd_data=[]
        st.session_state.generation_complete=True

    measurements= st.session_state.measurements
    if st.session_state.generation_complete and measurements:
        st.markdown("**Preview of Generated Data**")
        # Show partial time history for the first sensor
        first_data= measurements[0]["data"][:500]
        if first_data:
            df_sensor= pd.DataFrame(first_data)
            st.caption(f"Time History (Sensor {measurements[0]['sensorId']})")
            line_chart= alt.Chart(df_sensor).mark_line().encode(
                x=alt.X("time", title="Time (s)"),
                y=alt.Y("acceleration", title="Acceleration"),
                tooltip=["time","acceleration"]
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

        st.caption("**Sensor Positions**")
        df_pos= pd.DataFrame({
            "sensorId": [m["sensorId"] for m in measurements],
            "position": [m["position"] for m in measurements],
            "y": [0.5]* len(measurements)
        })
        scatter_chart= alt.Chart(df_pos).mark_point(size=100, shape="triangle").encode(
            x=alt.X("position", title="Position (m)"),
            y=alt.Y("y", title="", scale=alt.Scale(domain=[0,1])),
            tooltip=["sensorId","position"]
        ).properties(height=150)
        st.altair_chart(scatter_chart, use_container_width=True)
        st.caption(" | ".join([f"{m['sensorId']}: {m['position']:.2f} m" for m in measurements]))

    st.markdown("---")
    st.subheader("Analysis")
    analysis_method= st.selectbox("Choose Analysis Method", ("FDD","SSI"))

    def run_analysis():
        st.session_state.is_analyzing= True
        identified_modes, svd_data= perform_analysis(
            analysis_method, st.session_state.params, st.session_state.measurements
        )
        st.session_state.identified_modes= identified_modes
        st.session_state.svd_data= svd_data
        st.session_state.is_analyzing= False

    st.button("Perform Analysis", on_click=run_analysis, disabled=not st.session_state.generation_complete)

    identified_modes= st.session_state.identified_modes
    svd_data= st.session_state.svd_data
    if identified_modes:
        st.markdown(f"### Results ({analysis_method})")
        if analysis_method=="FDD" and svd_data:
            st.markdown("#### Singular Value Decomposition (SV1, SV2, SV3)")
            df_svd= pd.DataFrame(svd_data)
            max_x= 2.0* st.session_state.params["mode3"]["freq"]

            col_sv= st.columns(3)
            with col_sv[0]:
                chart_sv1= alt.Chart(df_svd).mark_line().encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv1", title="Amplitude"),
                    tooltip=["frequency","sv1"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv1, use_container_width=True)
            with col_sv[1]:
                chart_sv2= alt.Chart(df_svd).mark_line(color="red").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv2", title="Amplitude"),
                    tooltip=["frequency","sv2"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv2, use_container_width=True)
            with col_sv[2]:
                chart_sv3= alt.Chart(df_svd).mark_line(color="green").encode(
                    x=alt.X("frequency", title="Frequency (Hz)", scale=alt.Scale(domain=[0,max_x])),
                    y=alt.Y("sv3", title="Amplitude"),
                    tooltip=["frequency","sv3"]
                ).properties(height=300).interactive()
                st.altair_chart(chart_sv3, use_container_width=True)

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

        st.latex(
            r"""
            \text{MAC} =
            \frac{\left| \phi_i^T \phi_j \right|^2}
                {\left( \phi_i^T \phi_i \right) \left( \phi_j^T \phi_j \right)}
            """
        )

        st.markdown("#### Mode Shape Visualization")
        n_modes= len(identified_modes)
        col_vis= st.columns(n_modes)
        for i, mode_info in enumerate(identified_modes):
            with col_vis[i]:
                st.markdown(f"**Mode {mode_info['modeNumber']}**")
                df_mode= pd.DataFrame(mode_info["modeShapes"])

                needed_cols= {"position","trueShape","identifiedShape"}
                if not needed_cols.issubset(df_mode.columns) or df_mode.empty:
                    st.write("No shape data available for pinned-shape plots.")
                    continue

                # pinned-shape approach
                df_true_line= pd.concat([
                    pd.DataFrame({"position":[0],"trueShapeN":[0]}),
                    df_mode[["position","trueShape"]].rename(columns={"trueShape":"trueShapeN"}),
                    pd.DataFrame({"position":[params["bridgeLength"]], "trueShapeN":[0]})
                ], ignore_index=True)
                df_ident_line= pd.concat([
                    pd.DataFrame({"position":[0],"identifiedShapeN":[0]}),
                    df_mode[["position","identifiedShape"]].rename(columns={"identifiedShape":"identifiedShapeN"}),
                    pd.DataFrame({"position":[params["bridgeLength"]], "identifiedShapeN":[0]})
                ], ignore_index=True)

                line_true= alt.Chart(df_true_line).mark_line(
                    strokeDash=[5,3], color="gray"
                ).encode(
                    x=alt.X("position", title="Position (m)"),
                    y=alt.Y("trueShapeN", 
                            title="Normalized Amplitude",
                            scale=alt.Scale(domain=[-1.1,1.1]))
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
    st.subheader("Further Reading & Credits")
    st.markdown("""
    1. Brincker, R., & Ventura, C. (2015). *Introduction to Operational Modal Analysis*. Wiley.
    2. Peeters, B., & De Roeck, G. (2001). *Stochastic System Identification for Operational Modal Analysis: A Review*.
       Journal of Dynamic Systems, Measurement, and Control, 123(4), 659-667.
    3. Rainieri, C., & Fabbrocino, G. (2014). *Operational Modal Analysis of Civil Engineering Structures*. Springer.
    4. Au, S. K. (2017). *Operational Modal Analysis: Modeling, Bayesian Inference, Uncertainty Laws*. Springer.
    5. Brincker, R., Zhang, L., & Andersen, P. (2000). *Modal identification from ambient responses using frequency domain decomposition*.
       In Proceedings of the 18th International Modal Analysis Conference (IMAC), San Antonio, Texas.
    
    Thank you for using this interactive OMA tool. 
    We hope the refinements to the SSI indexing help you avoid out-of-range errors 
    and allow smooth demonstration of these methods.
    """)

###############################################################################
# (7) If run directly, launch main
###############################################################################
if __name__=="__main__":
    main()
