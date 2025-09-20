import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go
import time
from PIL import Image
import torch

# Import our inference tools
from src.inference import load_model, denoise_ecg_signal, DEVICE
from src.data_utils import TARGET_FS

# --- Page Configuration ---
st.set_page_config(
    page_title="Physics-Aware ECG Denoiser",
    page_icon="❤️",
    layout="wide",
)

# --- Model Loading ---
@st.cache_resource
def get_models():
    """Load and cache both models."""
    denoiser = load_model('models/ecg_denoiser_model.pth', DEVICE)
    classifier = None # Placeholder for the classifier model
    # Add try-except for robustness in case the file is missing
    try:
        from src.classifier_model import ECGClassifier
        classifier_model = ECGClassifier(num_classes=5)
        classifier_model.load_state_dict(torch.load('models/ecg_classifier_model.pth', map_location=DEVICE))
        classifier_model.to(DEVICE)
        classifier_model.eval()
        classifier = classifier_model
        print("Classifier model loaded successfully.")
    except Exception as e:
        print(f"Could not load classifier model: {e}")
    return denoiser, classifier

denoiser_model, classifier_model = get_models()

# --- Analysis & Plotting (Keep these functions as they are) ---
def analyze_ecg(signal, fs):
    """Calculate heart rate and rhythm status from R-peaks."""
    peaks, _ = find_peaks(signal, height=np.mean(signal) + np.std(signal), distance=fs*0.4)
    if len(peaks) < 2: return "N/A", "N/A", peaks
    rr_intervals = np.diff(peaks) / fs
    heart_rate = 60 / np.mean(rr_intervals)
    cov = np.std(rr_intervals) / np.mean(rr_intervals)
    rhythm_status = "Regular" if cov < 0.15 else "Irregular"
    return f"{heart_rate:.1f} bpm", rhythm_status, peaks

def classify_beats(signal, peaks, classifier, fs, window_size=128):
    """Classify each detected heartbeat."""
    if classifier is None or len(peaks) == 0:
        return {}
    
    beat_types = {0: 'Normal', 1: 'SVEB', 2: 'VEB', 3: 'Fusion', 4: 'Unknown'}
    predictions = []
    
    for p in peaks:
        start, end = p - window_size//2, p + window_size//2
        if start >= 0 and end < len(signal):
            beat_window = signal[start:end].astype(np.float32)
            with torch.no_grad():
                tensor_in = torch.from_numpy(beat_window).unsqueeze(0).unsqueeze(0).to(DEVICE)
                pred_label = torch.argmax(classifier(tensor_in), dim=1).item()
                predictions.append(beat_types.get(pred_label, 'Unknown'))
    
    # Return a count of each beat type found
    from collections import Counter
    return Counter(predictions)

def create_ecg_plot(signal, peaks, title, fs):
    """Create an interactive Plotly chart for the ECG signal."""
    time_axis = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=signal, mode='lines', name='ECG Signal', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=time_axis[peaks], y=signal[peaks], mode='markers', name='R-peaks', marker=dict(color='red', size=8, symbol='x')))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude (mV)", hovermode="x unified")
    return fig

# --- Main App UI ---
st.title("❤️ Physics-Aware ECG Denoiser & Diagnostic Tool")
st.markdown("An end-to-end AI pipeline to clean noisy ECG signals and provide diagnostic insights, designed for accessibility in rural healthcare.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Demo", "🔬 The Problem & Our Solution", "📈 Validation Results", "💻 The Technology"])

# --- TAB 1: Live Demo ---
with tab1:
    st.header("Transform Your Noisy ECG in Seconds")

    with st.sidebar:
        st.image("assets/header_image.png", use_container_width=True) # A nice header image
        st.title("Get Started")
        st.markdown(
            """
            1.  **Download the sample file** to see the app in action.
            2.  **Upload the sample file** (or your own single-column CSV ECG file).
            3.  Review the AI-powered denoising and automated diagnosis.
            """
        )
        
        try:
            with open("samples/sample_noisy_ecg.csv", "rb") as file:
                st.download_button(
                    label="Download Sample ECG (.csv)",
                    data=file,
                    file_name="sample_noisy_ecg.csv",
                    mime="text/csv"
                )
        except FileNotFoundError:
            st.error("Sample file not found on server.")

        st.markdown("---")
        uploaded_file = st.file_uploader("Upload your noisy ECG file", type=["csv", "txt"])
    
    # Main panel for displaying results
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            noisy_signal_np = df[0].to_numpy()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            noisy_signal_np = None
        
        if noisy_signal_np is not None:
            if len(noisy_signal_np) != 2048:
                st.warning(f"⚠️ Input signal has {len(noisy_signal_np)} samples, but the model is optimized for 2048. The signal will be padded or truncated.", icon="⚠️")
                padded_signal = np.zeros(2048)
                if len(noisy_signal_np) < 2048:
                    padded_signal[:len(noisy_signal_np)] = noisy_signal_np
                else:
                    padded_signal = noisy_signal_np[:2048]
                noisy_signal_np = padded_signal

            st.subheader("Denoising and Analysis in Progress...")
            
            with st.spinner('AI is cleaning the signal...'):
                denoised_signal_np = denoise_ecg_signal(noisy_signal_np, denoiser_model)
            
            hr_noisy, rhythm_noisy, peaks_noisy = analyze_ecg(noisy_signal_np, TARGET_FS)
            hr_denoised, rhythm_denoised, peaks_denoised = analyze_ecg(denoised_signal_np, TARGET_FS)
            
            with st.spinner('AI is classifying heartbeats...'):
                beat_counts = classify_beats(denoised_signal_np, peaks_denoised, classifier_model, TARGET_FS)

            st.success("Processing Complete!", icon="✅")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Noisy Signal")
                st.metric("Heart Rate", hr_noisy)
                st.metric("Rhythm", rhythm_noisy)
                st.plotly_chart(create_ecg_plot(noisy_signal_np, peaks_noisy, "Original Noisy ECG", TARGET_FS), use_container_width=True)
            
            with col2:
                st.subheader("AI Denoised & Analyzed Signal")
                st.metric("Heart Rate", hr_denoised)
                st.metric("Rhythm", rhythm_denoised)
                if beat_counts:
                    st.markdown("**Detected Beat Types:**")
                    st.json(beat_counts)
                st.plotly_chart(create_ecg_plot(denoised_signal_np, peaks_denoised, "AI Denoised ECG", TARGET_FS), use_container_width=True)
    else:
        st.info("Upload an ECG file or download and upload the sample to begin.")

# --- TAB 2: The Problem & Our Solution ---
with tab2:
    st.header("The Challenge: Unreliable Data in Critical Settings")
    st.markdown(
        """
        Modern electrocardiograms (ECGs) are vital for detecting cardiac issues, but their effectiveness hinges on signal quality. In many real-world settings, especially in under-resourced or rural areas, clinicians rely on low-cost, portable ECG devices. While accessible, these devices are highly susceptible to noise from muscle tremors, electrode motion, and baseline wander.

        **This is not a minor inconvenience; it's a critical diagnostic barrier.**
        - Research has shown that automated systems can miss up to **30% of acute heart attacks (STEMIs)** when the ECG signal is poor.
        - Field studies have uncovered significant rates of **undiagnosed atrial fibrillation (8% in one study)**, a condition easily detectable with a clean ECG but often hidden by noise.

        A noisy signal can lead to a missed diagnosis or a false alarm, both of which have serious consequences.
        """
    )

    st.header("Our Solution: A Physics-Aware AI Pipeline")
    st.markdown(
        """
        To address this, we built a novel end-to-end deep learning pipeline that doesn't just "clean" the signal, but does so in a way that respects the underlying physics of the heart.

        **1. The Denoiser:**
        A **1D U-Net** architecture takes the noisy signal and reconstructs a clean version. What makes it unique is our custom **physics-informed loss function**, which teaches the model not just to remove noise, but to preserve the essential diagnostic features of a real heartbeat:
        - **Gradient Loss:** Enforces the sharp, high-velocity spikes of the QRS complex.
        - **FFT Loss:** Ensures the output has a realistic frequency profile, matching the harmonic signature of a true ECG.

        **2. The Classifier:**
        A lightweight **1D Convolutional Neural Network (CNN)** takes the clean, denoised signal and performs beat-by-beat classification, identifying potential arrhythmias according to the AAMI standard.

        This two-stage approach ensures that the diagnostic model receives the highest quality data, dramatically improving its accuracy and reliability.
        """
    )

# --- TAB 3: Validation Results ---
with tab3:
    st.header("Proof of Impact: A Real-World Simulation")
    st.markdown(
        """
        To prove our system works, we conducted a rigorous test. We took a patient record from the MIT-BIH database that the AI had **never seen during training**. We then digitally added a severe level of noise to simulate a worst-case, low-quality recording (0 dB SNR). Finally, we ran this challenging signal through our full pipeline and measured the diagnostic accuracy.
        """
    )
    st.subheader("The Results: From Chaos to Clarity")
    st.markdown("The improvement in diagnostic accuracy is dramatic. We measured the F1-score, a key metric for classification performance:")
    
    result_data = {
        'Metric': ['Overall Accuracy', 'F1-Score (Ventricular Beats)', 'F1-Score (Supraventricular Beats)'],
        'On Noisy Signal': ['90.0%', '0.80', '0.34'],
        'On Denoised Signal (Our Tool)': ['96.0%', '**0.97**', '**0.67**']
    }
    st.table(pd.DataFrame(result_data))
    
    st.markdown(
        """
        **Key Takeaway:** Our denoising step increased the F1-score for detecting critical **Ventricular arrhythmias from 0.80 to 0.97** and nearly doubled the score for Supraventricular arrhythmias.
        """
    )

    st.subheader("Visualizing the Improvement: Confusion Matrices")
    st.markdown("A confusion matrix shows what a model predicted vs. the actual truth. A perfect model has a bright line from the top-left to the bottom-right. Notice how the 'Denoised' matrix is significantly cleaner and more accurate than the 'Noisy' one.")
    
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open('results/confusion_matrix_noisy.png'), caption="Confusion Matrix on Noisy Data")
        with col2:
            st.image(Image.open('results/confusion_matrix_denoised.png'), caption="Confusion Matrix on Denoised Data")
    except FileNotFoundError:
        st.warning("Validation images not found. Please run 'src/validate_end_to_end.py' to generate them.")

# --- TAB 4: The Technology ---
with tab4:
    st.header("Built with Open-Source Tools")
    st.markdown(
        """
        This project was built entirely with free, open-source tools, ensuring its accessibility and reproducibility.

        - **Backend & Modeling:** Python, PyTorch, Scikit-learn, NumPy, SciPy
        - **Data Source:** PhysioNet (MIT-BIH Arrhythmia & Noise Stress Test Databases)
        - **Frontend:** Streamlit
        - **Training Environment:** Google Colab (leveraging free T4 GPUs)

        The complete source code, including training and validation scripts, is available on our GitHub repository.
        """
    )
    st.link_button("View on GitHub", "https://github.com/Mohan-CAS-and-hackathons/ecg-denoiser-hackathon")