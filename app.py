# --- START OF FINAL FILE app.py ---
import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go
import time
import os

# Import our inference tools
from src.inference import load_model, denoise_ecg_signal, DEVICE
from src.data_utils import TARGET_FS # Use the same target frequency

# --- Page Configuration ---
st.set_page_config(
    page_title="Physics-Aware ECG Denoiser",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
@st.cache_resource
def get_model():
    """Load and cache the denoising model."""
    # Ensure the model file is found
    model_path = 'ecg_denoiser_model.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please make sure it's in the root directory.")
        return None
    return load_model(model_path, DEVICE)

model = get_model()

# --- Analysis & Plotting (These functions remain unchanged) ---
def analyze_ecg(signal, fs):
    """Calculate heart rate and rhythm status from R-peaks."""
    peaks, _ = find_peaks(signal, height=np.mean(signal) + 0.5 * np.std(signal), distance=fs*0.4)
    if len(peaks) < 2: return "N/A", "N/A", peaks
    rr_intervals = np.diff(peaks) / fs
    heart_rate = 60 / np.mean(rr_intervals)
    cov = np.std(rr_intervals) / np.mean(rr_intervals)
    rhythm_status = "Regular" if cov < 0.15 else "Irregular"
    return f"{heart_rate:.1f} bpm", rhythm_status, peaks

def create_ecg_plot(signal, peaks, title, fs):
    """Create an interactive Plotly chart for the ECG signal."""
    time_axis = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=signal, mode='lines', name='ECG Signal', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time_axis[peaks], y=signal[peaks], mode='markers', name='R-peaks', marker=dict(color='red', size=8, symbol='x')))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude (mV)", hovermode="x unified")
    return fig

# --- Main App UI ---
st.title("❤️ Physics-Aware ECG Denoiser")
st.markdown("An accessible AI tool to clean noisy ECG signals from low-cost devices, enabling better cardiac diagnostics in rural and under-resourced areas.")
st.markdown("---")

# Sidebar for file upload and instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        1.  **Download the sample file** to test the app.
        2.  **Upload the sample file** (or your own single-column CSV ECG file).
        3.  Review the AI-powered denoising and diagnostics.
        """
    )
    
    # Add a button to download the pre-generated sample file
    try:
        with open("sample_noisy_ecg.csv", "rb") as file:
            st.download_button(
                label="Download Sample ECG (.csv)",
                data=file,
                file_name="sample_noisy_ecg.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.error("Sample file 'sample_noisy_ecg.csv' not found.")

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your noisy ECG file", type=["csv", "txt"])

# Main panel for displaying results
if uploaded_file is not None:
    if model is None:
        st.stop() # Stop execution if the model failed to load

    try:
        df = pd.read_csv(uploaded_file, header=None)
        noisy_signal_np = df[0].to_numpy()
    except Exception as e:
        st.error(f"Error reading file. Please ensure it is a single-column CSV. Details: {e}")
        noisy_signal_np = None
    
    if noisy_signal_np is not None:
        # Pad or truncate the signal to the required model input length (2048)
        original_length = len(noisy_signal_np)
        if original_length != 2048:
            st.warning(f"⚠️ Input signal has {original_length} samples. The model requires 2048 samples (approx. 8.2s). The signal will be padded or truncated.")
            if original_length < 2048:
                padded_signal = np.zeros(2048)
                padded_signal[:original_length] = noisy_signal_np
                noisy_signal_np = padded_signal
            else:
                noisy_signal_np = noisy_signal_np[:2048]

        st.header("Denoising Results")
        
        with st.spinner('AI is cleaning the signal...'):
            time.sleep(1) # Simulate a bit of work for a better UX
            denoised_signal_np = denoise_ecg_signal(noisy_signal_np, model)
        
        # Analyze both signals
        hr_noisy, rhythm_noisy, peaks_noisy = analyze_ecg(noisy_signal_np, TARGET_FS)
        hr_denoised, rhythm_denoised, peaks_denoised = analyze_ecg(denoised_signal_np, TARGET_FS)
        
        # Display metrics side-by-side
        st.subheader("Diagnostic Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Heart Rate", hr_noisy)
            st.metric("Original Rhythm", rhythm_noisy)
        
        with col2:
            st.metric("Denoised Heart Rate", hr_denoised)
            st.metric("Denoised Rhythm", rhythm_denoised)

        st.markdown("---")
        
        # Display plots
        st.subheader("Signal Visualization")
        st.plotly_chart(create_ecg_plot(noisy_signal_np, peaks_noisy, "Original Noisy ECG", TARGET_FS), use_container_width=True)
        st.plotly_chart(create_ecg_plot(denoised_signal_np, peaks_denoised, "AI Denoised ECG", TARGET_FS), use_container_width=True)
else:
    st.info("Please upload an ECG file to begin. You can use the sample file available for download in the sidebar.")
# --- END OF FINAL FILE app.py ---