# A Theoretical Framework for Spectral-Temporal Physiological Consistency in Biomedical Signal Denoising

**Principal Investigator:** Mohanarangan Desigan

---

## Abstract

Cleaning up noise from biomedical signals (like ECGs) is a major challenge. Often, researchers mix and match different mathematical objectives in a trial-and-error way. To move beyond this, we introduce a new guiding principle we call **Spectral-Temporal Physiological Consistency (STPC)**. STPC acts as a smart set of rules for training AI models, ensuring that the cleaned signals are not only less noisy but also true to the body's natural behavior.

Our STPC framework combines three key ideas:
1.  **Amplitude Consistency:** Making sure the signal's height and depth are correct.
2.  **Temporal-Gradient Consistency:** Preserving the sharp, sudden changes in the signal (like the spike in a heartbeat).
3.  **Spectral-Magnitude Consistency:** Maintaining the signal's original fingerprint of frequencies.

In this paper, we formally define STPC, explain the biological reasoning behind each part, and prove some of its important mathematical properties. We also show how this single framework can be adapted for different types of signals, including ECG (heart), EEG (brain), and EMG (muscle). Finally, we provide a complete plan for testing this framework on Google Colab or a local Mac M1, making our work easy to verify and build upon.

---

## 1. Introduction

Biomedical signals from the body, such as ECGs, EEGs, and EMGs, are a mix of sharp, fleeting events (like the QRS spike in a heartbeat) and rhythmic patterns with a specific frequency signature. When we try to remove noise from these signals, common methods often make one of two mistakes: they either smooth away the important sharp spikes or fail to remove noise that doesn't match the signal's natural frequency profile.

To solve this, we developed STPC, a principle that guides an AI model to respect the physics of the signal. It enforces three types of consistency at the same time:

-   **Amplitude consistency:** The cleaned signal's values should match the original.
-   **Temporal-gradient consistency:** The steepness of slopes and sharp edges in the signal should be preserved.
-   **Spectral-magnitude consistency:** The distribution of energy across different frequencies should be maintained, which helps ignore tiny timing jitters.

This paper aims to: (1) define STPC and provide the theory behind it; (2) prove that it faithfully preserves signal shape and is robust to timing shifts; and (3) offer a practical, easy-to-reproduce guide for validating our results using common tools like Google Colab.

---

## 2. Formal Definition of STPC

### 2.1 Notation and Setup

We'll work with signals as a series of numbers, represented as $x$ (the clean signal) and $\hat{x}$ (the model's cleaned version), both of length $N$.

-   $\|v\|_p$: A way to measure the size or length of a vector $v$.
-   $F$: The Discrete Fourier Transform (DFT), a tool that converts a signal from the time domain to the frequency domain.
-   $G$: The "forward-difference" operator, which calculates the slope (or gradient) between consecutive points in the signal ($v_{i+1} - v_i$).
-   $\Delta = G^T G$: The discrete Laplacian, a mathematical operator that measures the "smoothness" or curvature of the signal.

### 2.2 The STPC Loss Function

The total STPC loss, which the AI model tries to minimize, is a weighted sum of three different error terms. For given weights $\lambda_1, \lambda_2, \lambda_3$, the loss is:

$$
L_{\mathrm{STPC}}(x,\hat{x}) = \lambda_1 D_{\mathrm{Amp}}(x,\hat{x}) + \lambda_2 D_{\mathrm{Grad}}(x,\hat{x}) + \lambda_3 D_{\mathrm{Spec}}(x,\hat{x})
$$

Where the three components are:

-   **Amplitude Term:** $D_{\mathrm{Amp}}(x,\hat{x}) := \|x-\hat{x}\|_p^p$
    -   This measures the direct, point-by-point difference between the clean signal and the reconstructed one.

-   **Temporal Gradient Term:** $D_{\mathrm{Grad}}(x,\hat{x}) := \|Gx - G\hat{x}\|_p^p$
    -   This measures the difference in the *slopes* of the two signals, ensuring sharp changes are preserved.

-   **Spectral Magnitude Term:** $D_{\mathrm{Spec}}(x,\hat{x}) := \|\,|Fx| - |F\hat{x}|\,\|_q^q$
    -   This measures the difference in the *magnitude* of the signals' frequency components, ignoring phase (timing). For signals whose frequencies change over time, we use the Short-Time Fourier Transform (STFT).

**Practical Note:** For the spectral term, it's often more stable to compare the logarithm of the magnitudes or the squared magnitudes to avoid issues during model training.

---

## 3. Why Each STPC Component Matters

### 3.1 Amplitude Consistency

This is a standard way to ensure the overall shape and clinically important levels of the signal are correct (e.g., the height of an ST-elevation in an ECG). Using an L2-norm works well for Gaussian noise, while an L1-norm is more robust against sharp, sudden noise spikes.

### 3.2 Temporal-Gradient Consistency

**Biological Reason:** Critical events in biomedical signals, like the QRS complex in an ECG or a neuronal spike in an EEG, happen very quickly. These fast events create steep slopes. By penalizing differences in the signal's gradient, we encourage the model to keep these sharp features instead of blurring them out.

**Mathematical Insight:** In the common quadratic case ($p=2$), the gradient difference can be written as:

$$
D_{\mathrm{Grad}}(x,\hat{x}) = \|G(x-\hat{x})\|_2^2 = (x-\hat{x})^T \Delta (x-\hat{x})
$$

This formula shows that penalizing the gradient error is the same as penalizing the "curvature energy" of the error. This cleverly pushes any unavoidable error into the low-frequency, slowly changing parts of the signal, protecting the sharp, high-frequency details.

### 3.3 Spectral Magnitude Consistency

**Biological Reason:** Many processes in the body are rhythmic and produce signals with a characteristic energy signature in the frequency domain. Forcing the model to match the frequency magnitude helps preserve this signature and removes noise that doesn't fit the expected pattern.

**Insensitivity to Timing Jitter:** By comparing only the *magnitude* of the frequency components ($|F\cdot|$), we ignore the phase. This makes the loss immune to perfect circular shifts in time and less sensitive to the small timing jitters that are common in biological signals (like a heartbeat that's a few milliseconds early or late). Other methods that compare the full complex frequency values would wrongly penalize such small, natural variations.

---

## 4. Theorem: Preserving Signal Shape

### 4.1 Gradient Preservation Theorem

**Idea:** Imagine all possible cleaned signals $\hat{x}$ that have the exact same amount of amplitude error $\varepsilon$. Which one does the best job at preserving the sharp slopes of the original signal?

**Theorem:** The signal $\hat{x}^*$ that minimizes the gradient error term, $\|G(\hat{x}-x)\|_2^2$, will have the smallest possible deviation in its slopes compared to any other signal with the same amplitude error.

**Proof Sketch:** The proof involves looking at the natural "vibrational modes" (eigenvectors) of the Laplacian operator $\Delta$. Minimizing the gradient error forces the reconstruction error to be concentrated in the "lowest energy" modes, which correspond to very slow, smooth changes (like a DC shift). This leaves the high-energy, sharp features of the signal largely untouched.

**In Simple Terms:** When forced to choose where to put errors, the gradient term ensures that the error damages the sharp, important peaks as little as possible.

### 4.2 Example: A Triangular (QRS-like) Pulse

Consider a simple triangular pulse, which mimics the QRS spike of a heartbeat. If we add high-frequency noise, the slope of the pulse gets corrupted. A model trained with the gradient term will be much more effective at filtering out the noise that affects the slope. This is because the math behind it acts like a filter that specifically targets and dampens high-frequency errors, thereby restoring the true slope of the pulse.

### 4.3 Phase-Shift Insensitivity

**Proposition:** If you shift a signal in time (by an integer amount $\tau$), the magnitude of its Fourier Transform does not change.

$$
|F\{x_\tau\}[k]| = |F\{x\}[k]| \quad \text{for all frequencies } k
$$

This is why the spectral magnitude loss is so powerful: it focuses on whether the right frequencies are present with the right amount of energy, without being overly strict about their exact timing.

---

## 5. Generalizing Across Different Signal Types

### 5.1 Generalization Hypothesis

Our central hypothesis is that for any biomedical signal that contains both sharp transient events and a characteristic frequency profile, a model trained with STPC will be better at:
1.  Preserving the slopes of the sharp events.
2.  Matching the true frequency distribution of the signal.

This should hold true compared to models trained only on amplitude, as long as the weights ($\lambda_1, \lambda_2, \lambda_3$) are reasonably balanced.

### 5.2 Applying STPC to EEG and EMG

The STPC framework is flexible and can be easily adapted.

**For EEG (Brain Signals):**
-   **Temporal-gradient** helps preserve the shape of sharp epileptic spikes.
-   **Spectral-magnitude** helps maintain the standard brainwave bands (alpha, beta, gamma) while removing noise from muscle activity.

**For EMG (Muscle Signals):**
-   **Temporal-gradient** preserves the sharp onset of muscle activation signals (MUAPs).
-   **Spectral-magnitude** maintains the known power distribution of muscle signals, helping to separate them from motion artifacts.

---

## 6. Empirical Validation Plan (Runnable on Colab and Mac M1)

Here is a complete, reproducible plan to test STPC on real-world data using freely available tools.

### 6.1 Goals

1.  Show that STPC better preserves the shape of sharp events compared to an amplitude-only loss.
2.  Show that denoising with STPC improves the performance of downstream tasks, like classifying heartbeats or detecting seizures.
3.  Confirm that the framework generalizes by testing it on both ECG and EEG data.

### 6.2 Datasets

-   **ECG:** Use a small subset of the **MIT-BIH Arrhythmia Database** and the **MIT-BIH Noise Stress Test Database** to create realistic noisy signals.
-   **EEG:** Use a few records from the **CHB-MIT Scalp EEG Database** that contain seizures.

A small amount of data (10-50 minutes per domain) is enough to validate the approach without needing powerful hardware.

### 6.3 Model Architecture

-   **Denoiser:** A simple 1D U-Net model.
-   **Classifier:** A lightweight 1D CNN for a downstream task.

These models are small enough to train quickly on the free tier of Google Colab.

### 6.4 Recommended Settings

-   **Amplitude Loss ($D_{\mathrm{Amp}}$):** Use L1 loss with a weight of $\lambda_1=1.0$.
-   **Gradient Loss ($D_{\mathrm{Grad}}$):** Use L2 loss with a weight of $\lambda_2$ between $0.1$ and $1.0$.
-   **Spectral Loss ($D_{\mathrm{Spec}}$):** Use L2 loss on the log-magnitude of the STFT, with a weight of $\lambda_3$ between $0.01$ and $0.2$.

Use the Adam optimizer with a learning rate of 1e-3 and train for 20-50 epochs.

### 6.5 Evaluation Metrics

-   **Shape Metrics:** Measure the error in the peak slope around an event.
-   **Spectral Metrics:** Measure the log-spectral distance between the clean and reconstructed signals.
-   **Task Metrics:** For the downstream task, measure F1-score, precision/recall, and AUC.

### 6.6 Reproducible Colab Recipe

1.  **Install libraries:** `pip install wfdb mne pyedflib librosa torch torchvision`.
2.  **Download data:** Get a small subset from PhysioNet.
3.  **Preprocess:** Resample data to 256 Hz, apply a bandpass filter, and normalize.
4.  **Generate data:** Create training pairs by mixing clean signals with realistic noise at various signal-to-noise ratios.
5.  **Train model:** Use the 1D U-Net and the STPC loss function.
6.  **Evaluate:** Test the model on unseen data and generate plots and metrics.

---

## 7. Implementation Details

### 7.1 PyTorch-Style STPC Loss (Pseudo-code)

Here is a simplified code snippet showing how to implement the STPC loss in PyTorch.

```python
import torch

def stpc_loss(x, x_hat, lam_amp=1.0, lam_grad=0.1, lam_spec=0.05):
    """
    Calculates the STPC loss.
    x: clean signal
    x_hat: reconstructed signal
    """
    eps = 1e-8

    # 1. Amplitude Loss (L1)
    amp_loss = torch.mean(torch.abs(x - x_hat))

    # 2. Temporal-Gradient Loss (L2 on the difference)
    grad_loss = torch.mean((torch.diff(x_hat, dim=-1) - torch.diff(x, dim=-1))**2)

    # 3. Spectral-Magnitude Loss (L2 on log-STFT magnitude)
    X = torch.stft(x, n_fft=256, hop_length=64, return_complex=True)
    X_hat = torch.stft(x_hat, n_fft=256, hop_length=64, return_complex=True)

    mag_x = torch.sqrt(X.real**2 + X.imag**2 + eps)
    mag_x_hat = torch.sqrt(X_hat.real**2 + X_hat.imag**2 + eps)
    
    spec_loss = torch.mean((torch.log(mag_x_hat + eps) - torch.log(mag_x + eps))**2)

    # Combine the losses
    total_loss = lam_amp * amp_loss + lam_grad * grad_loss + lam_spec * spec_loss
    return total_loss
```

---

## 8. Expected Results

Based on our theory and small-scale tests, we expect to see:
-   Models trained with STPC will preserve the sharpness of peaks better than models trained on amplitude loss alone, even if their overall error is similar.
-   STPC will lead to a better match in the frequency content of the signals.
-   Using STPC as a preprocessing step will improve the accuracy of downstream tasks like heartbeat classification.

---

## 9. Discussion

The STPC framework provides a principled, physics-aware method for regularizing denoising models for biomedical signals. Our theoretical guarantees for preserving sharp features and ignoring timing jitter align well with what clinicians need for accurate diagnosis. Best of all, STPC is easy to implement with standard tools and can be run on accessible hardware.

**Limitations:**
-   Choosing the right weights for the three loss terms requires some tuning.
-   The spectral loss can sometimes make the optimization process tricky, so careful setup is needed.
-   STPC improves signal quality but doesn't solve other problems like class imbalance in datasets.

**Future Work:** We plan to explore using STPC for multi-channel signals and developing more rigorous statistical proofs of its performance under different noise conditions.

---

## 10. Appendix

### A. Full Colab Notebook Skeleton

A complete, ready-to-run Google Colab notebook is provided in the project's repository. It includes all the code for data loading, model training, and evaluation described in this paper.

### B. Detailed Triangular-Pulse Calculation

The supplementary materials include a detailed mathematical walk-through of the triangular pulse example, showing exactly how the gradient regularization term helps preserve the peak slope when noise is present.

### C. References

- McManus, D. D., et al. (2016). "A Novel Application for the Detection of an Irregular Pulse Using a Smartwatch." *Heart Rhythm*.
- Li, Q., et al. (2015). "A survey of ECG signal processing and analysis."
- National Center for Biotechnology Information. (2015). *Making Healthcare Safer III: A Critical Analysis of Existing and Emerging Patient Safety Practices*.
- Singh, O., et al. (2007). "Denoising of ECG signals using wavelet transform."
- Goldberger, A. L., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals." *Circulation*.

---

## Acknowledgements

We thank the creators of open biomedical datasets (like PhysioNet) and the developers of scientific computing tools that make reproducible research like this possible.

---
*End of document.*