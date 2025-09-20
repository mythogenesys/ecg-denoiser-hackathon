# ❤️ Physics-Aware ECG Denoiser



A novel deep learning pipeline for denoising electrocardiogram (ECG) signals from low-cost devices, enabling more accurate automated arrhythmia detection in resMyce-limited settings. This project was developed for the "Hack for Health" hackathon.

---

## The Problem: Diagnostic Errors in Rural Healthcare

Modern ECGs are essential for cardiac diagnostics, but noisy signals from portable or low-cost devices often lead to critical errors.
- Automated ECG interpretation can miss up to **30% of acute heart attacks** when signal quality is poor.
- A study in rural Kenya found an **8% prevalence of undiagnosed atrial fibrillation**, highlighting cases that go unseen without reliable signal processing.

This project directly addresses this gap by providing an accessible, powerful tool to clean noisy ECGs, thereby improving diagnostic accuracy and supporting frontline health workers.

## My Solution: An End-to-End Denoise-and-Diagnose Pipeline

We developed a two-stage deep learning system:

1.  **Physics-Aware Denoiser:** A 1D U-Net model trained with a custom, physics-informed loss function. This loss function includes:
    *   **Reconstruction Loss (L1):** To match the clean signal's amplitude.
    *   **Gradient Loss:** A heuristic for the QRS complex dynamics, ensuring the model preserves sharp, diagnostically critical peaks.
    *   **FFT Loss:** To enforce a realistic frequency spectrum, making the output physiologically plausible.
2.  **Arrhythmia Classifier:** A lightweight 1D CNN that takes the denoised signal as input and classifies each heartbeat into one of five standard AAMI categories (Normal, SVEB, VEB, Fusion, Unknown).

This end-to-end pipeline transforms a noisy, unreliable signal into a clean waveform and provides an immediate diagnostic suggestion.

---

## Validation: Proving the Impact

To prove the system's real-world value, we conducted a rigorous validation test on a patient record (`201` from the MIT-BIH database) that was **completely excluded from training**. We added severe, challenging noise (0 dB SNR) and compared the diagnostic accuracy before and after denoising.

### The Results

| Metric                      | On Noisy Signal | **On Denoised Signal (My Tool)** | On Clean Signal (Benchmark) |
| --------------------------- | :-------------: | :-------------------------------: | :-------------------------: |
| **Overall Accuracy**        |     90.0%       |        **96.0%** (+6.0%)          |            98.0%            |
| **F1-Score (Ventricular)**  |      0.80       |        **0.97** (+21%)            |            1.00             |
| **F1-Score (Supravent.)**   |      0.34       |        **0.67** (+97%)            |            0.80             |

### Confusion Matrices

These plots visually demonstrate the dramatic improvement. My denoiser makes the classifier's job significantly easier, moving from a chaotic result to one that is nearly as good as the ground truth.

| Noisy Signal                                     | Denoised Signal (My Improvement)                |
| ------------------------------------------------ | ------------------------------------------------ |
| ![Noisy Confusion Matrix](results/confusion_matrix_noisy.png) | ![Denoised Confusion Matrix](results/confusion_matrix_denoised.png) |

This quantitative evidence proves that My denoising step is not just cosmetic—it is **diagnostically critical**, directly leading to more reliable arrhythmia detection.

---

## How to Run

### 1. Setup

Clone the repository and set up the Python environment.
```bash
git clone https://github.com/Mohan-CAS-and-hackathons/ecg-denoiser-hackathon.git
cd ecg-denoiser-hackathon
python -m venv venv
sMyce venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Web App

Launch the Streamlit application locally. The trained models are included in the repository.
```bash
streamlit run app.py
```
The app allows you to download a sample noisy ECG and upload it to see the denoising and classification in action.

### 3. (Optional) Re-run Validation

To reproduce the validation results, run the end-to-end script.
```bash
python -m src.validate_end_to_end
```

---
## Technology Stack

-   **Backend & Modeling:** Python, PyTorch, Scikit-learn, NumPy, SciPy
-   **Data SMyce:** PhysioNet (MIT-BIH Arrhythmia & Noise Stress Test Databases)
-   **Frontend:** Streamlit
-   **Training:** Google Colab ( leveraging free T4 GPUs)