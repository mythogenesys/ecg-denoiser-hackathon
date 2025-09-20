# Physics-Aware Deep Learning for Robust ECG Denoising and Arrhythmia Detection in Low-Resource Settings

**Author:** Mohanarangan Desigan  
*Independent Researcher, Project for "Hack for Health"*

---

## Abstract

The diagnostic utility of electrocardiograms (ECGs) in remote and low-resource settings is often compromised by significant noise contamination inherent to portable, low-cost acquisition devices. This degradation of signal quality can mask critical arrhythmic events, leading to diagnostic errors. To address this challenge, we propose a novel, two-stage deep learning pipeline that first denoises the ECG signal using a physics-aware model and then performs automated arrhythmia classification. The denoising stage employs a 1D U-Net architecture trained with a composite, physics-informed loss function that includes L1 reconstruction, temporal gradient, and frequency-domain (FFT) components. This approach encourages the model to generate outputs that are not only clean but also physiologically plausible, preserving the sharp morphology and characteristic frequency spectrum of a true ECG. The second stage uses a lightweight 1D Convolutional Neural Network (CNN) to classify individual heartbeats from the denoised signal into five standard AAMI categories. To validate our system, we conducted an end-to-end simulation on an unseen patient record from the MIT-BIH Arrhythmia Database, artificially corrupted with severe noise (0 dB SNR). The results demonstrate that our denoising pre-processing step dramatically improves diagnostic accuracy, increasing the F1-score for detecting Ventricular Ectopic Beats from 0.80 on the noisy signal to 0.97, closely approaching the 1.00 score achieved on the clean ground truth. This work demonstrates the tangible value of physics-aware deep learning in creating accessible, reliable, and impactful diagnostic tools for global health.

---

## 1. Introduction

The electrocardiogram (ECG) is a cornerstone of cardiovascular medicine, providing a non-invasive window into the electrical activity of the heart. While hospital-grade 12-lead ECG systems are the gold standard, the proliferation of low-cost, portable, and wearable ECG devices has opened new avenues for remote patient monitoring and point-of-care diagnostics, particularly in underserved and rural areas. However, the clinical utility of these accessible devices is frequently hampered by a low signal-to-noise ratio (SNR). Noise artifacts stemming from muscle contraction (EMG), electrode motion, and baseline wander can obscure or mimic critical diagnostic features, leading to a high rate of interpretation errors by both human clinicians and automated algorithms. Studies have indicated that poor signal quality can cause automated systems to miss as many as 30% of acute myocardial infarctions [3], representing a significant barrier to effective care.

Traditional denoising methods, such as wavelet transforms and adaptive filtering, often struggle with the non-stationary nature of ECG noise and can inadvertently smooth over or distort diagnostically vital high-frequency components like the QRS complex [4]. While recent deep learning approaches have shown promise, many operate as "black boxes," and their outputs may not always conform to physiological reality.

This paper introduces a novel, end-to-end deep learning pipeline designed to address these challenges. Our primary contribution is a **Physics-Aware Denoising Model** that leverages a composite loss function to imbue the network with domain-specific knowledge of ECG morphology and dynamics. We hypothesize that by pre-processing a noisy signal with this denoiser, we can dramatically improve the performance of a subsequent automated arrhythmia classification model, thereby creating a robust and reliable diagnostic tool suitable for low-resource settings. We validate this hypothesis through a rigorous simulation of a real-world clinical scenario.

---

## 2. Methods

Our system is composed of two primary deep learning models—a denoiser and a classifier—and a data pipeline built upon publicly available datasets.

### 2.1. Datasets

We utilized two standard databases from PhysioNet [5]:
*   **MIT-BIH Arrhythmia Database:** This database contains 48 half-hour, dual-channel ambulatory ECG recordings from 47 subjects, with beat-by-beat annotations provided by expert cardiologists. It served as the source for our "clean" ECG signals and ground-truth diagnostic labels.
*   **MIT-BIH Noise Stress Test Database (NSTDB):** This database provides recordings of common noise artifacts, including muscle artifact (MA), electrode motion (EM), and baseline wander (BW), which were used to synthesize realistic noisy training data.

### 2.2. Stage 1: The Physics-Aware Denoiser

*   **Architecture:** We implemented a 1D U-Net architecture, which is well-suited for signal-to-signal tasks due to its symmetric encoder-decoder structure and use of skip connections. This design allows the model to capture both low-frequency contextual information (in its deep layers) and high-frequency temporal details (via skip connections), which is essential for reconstructing the full P-QRS-T waveform.

*   **Physics-Informed Loss Function:** The model was trained to minimize a composite loss function, $L_{\text{total}}$, defined as a weighted sum of three components:
    $$L_{\text{total}} = w_{\text{recon}} L_{\text{recon}} + w_{\text{grad}} L_{\text{grad}} + w_{\text{fft}} L_{\text{fft}}$$
    1.  **Reconstruction Loss ($L_{\text{recon}}$):** Standard L1 (Mean Absolute Error) loss between the denoised output and the clean target. This encourages amplitude fidelity.
    2.  **Gradient Loss ($L_{\text{grad}}$):** L1 loss between the temporal first derivatives of the output and target. This serves as a heuristic for the high-velocity dynamics of the QRS complex, penalizing blurred or smoothed peaks.
    3.  **Frequency-Domain Loss ($L_{\text{fft}}$):** L1 loss between the magnitudes of the Fast Fourier Transform (FFT) of the output and target. This enforces a physiologically plausible power spectrum, characteristic of the quasi-periodic nature of ECG signals.

*   **Training:** The U-Net was trained for 50 epochs using the AdamW optimizer. Training data consisted of 2048-sample segments generated on-the-fly by mixing clean ECG signals with noise from the NSTDB at random SNRs ranging from -3 dB to +12 dB.

### 2.3. Stage 2: The Arrhythmia Classifier

*   **Architecture:** A lightweight 1D Convolutional Neural Network (CNN) was designed to classify individual heartbeats. The model consists of two convolutional blocks followed by a fully connected head with dropout for regularization.

*   **Data Preparation:** Over 100,000 individual heartbeats were extracted from the MIT-BIH Arrhythmia Database. Each beat was represented as a 128-sample window centered on its annotated R-peak. Beats were mapped to five standard AAMI classes: Normal (N), Supraventricular Ectopic (S), Ventricular Ectopic (V), Fusion (F), and Unknown (Q).

*   **Training:** The classifier was trained for 20 epochs using the Adam optimizer and Cross-Entropy Loss on an 80/20 train/validation split of the extracted beats.

### 2.4. End-to-End Validation Protocol

To simulate a real-world scenario, we selected a patient record (`201`) that was entirely excluded from the training sets of both models. We created a "highly noisy" version of this entire record by adding muscle artifact noise to achieve a challenging SNR of 0 dB. We then performed beat-by-beat classification on three versions of the signal: (i) the clean, ground-truth signal, (ii) the highly noisy signal, and (iii) the noisy signal after being processed by our trained denoiser. Performance was evaluated using accuracy and the F1-score.

---

## 3. Results

The validation protocol yielded a clear and quantitative demonstration of our system's efficacy. The performance of the arrhythmia classifier was measured on all three signal conditions, with the results summarized in Table 1 and visualized as confusion matrices in Figure 1.

| Metric (F1-Score)        | On Noisy Signal | **On Denoised Signal** | On Clean Signal |
| :----------------------- | :-------------: | :--------------------: | :-------------: |
| Class N (Normal)         |      0.95       |        **0.98**        |      0.99       |
| Class S (Supravent.)     |      0.34       |        **0.67**        |      0.80       |
| Class V (Ventricular)    |      0.80       |        **0.97**        |      1.00       |
| **Overall Accuracy**     |     90.0%       |        **96.0%**       |      98.0%      |
*Table 1: Comparison of diagnostic F1-scores and accuracy across signal conditions on the unseen test record.*

On the highly noisy signal, the classifier's performance was significantly degraded, particularly for non-normal beats. The F1-score for detecting Supraventricular ('S') beats was only 0.34. After processing the signal with our physics-aware denoiser, the performance improved dramatically across all classes. Most notably, the F1-score for Ventricular ('V') beats surged from 0.80 to 0.97, and the score for 'S' beats nearly doubled to 0.67. The overall accuracy increased by 6 percentage points. The performance on the denoised signal closely approached the benchmark performance on the perfectly clean ground-truth signal.

The confusion matrices in Figure 1 provide a visual representation of this improvement. On the noisy signal (Fig. 1a), there is significant confusion, with many 'S' and 'V' beats being misclassified as 'N'. On the denoised signal (Fig. 1b), these off-diagonal errors are drastically reduced, and the matrix more closely resembles that of the clean signal (Fig. 1c).

*<p align="center">Figure 1: Confusion matrices for the arrhythmia classifier on (a) the noisy signal, (b) the denoised signal, and (c) the clean ground-truth signal.</p>*
| (a) Noisy Signal | (b) Denoised Signal | (c) Clean Signal |
| :---: | :---: | :---: |
| ![Noisy](results/confusion_matrix_noisy.png) | ![Denoised](results/confusion_matrix_denoised.png) | ![Clean](results/confusion_matrix_cleanGT.png) |

---

## 4. Discussion and Future Work

The results of our end-to-end validation robustly support our central hypothesis: a denoising pre-processing step, particularly one guided by physiological principles, can substantially improve the accuracy of automated ECG diagnostics. The dramatic increase in F1-scores for arrhythmic beats after denoising demonstrates that our U-Net model is not merely smoothing the signal, but is successfully restoring diagnostically critical information that was lost in the noise.

The choice of a physics-informed loss function was central to this success. By explicitly penalizing deviations in the signal's gradient and frequency spectrum, we guided the model to learn a representation that is consistent with the known biophysical properties of cardiac electrical activity. This represents a step away from "black box" models and towards more interpretable and reliable systems.

While our system shows significant promise, there are limitations and avenues for future work. The classifier's performance on rare classes (like Fusion beats) is limited by the imbalanced nature of the dataset. Future iterations could employ data augmentation or advanced sampling techniques to address this. The ultimate extension of this work would be to implement a task-oriented training scheme, where the denoiser and classifier are trained jointly, optimizing the denoising process not just for signal fidelity, but for maximizing the accuracy of the downstream diagnostic task.

In conclusion, our two-stage, physics-aware pipeline provides a powerful and accessible framework for improving the reliability of ECG diagnostics in low-resource settings. By leveraging deep learning to enhance, rather than replace, the diagnostic process, we can create tools that empower frontline health workers, reduce diagnostic errors, and ultimately improve patient outcomes.

---

## 5. References

[1] McManus, D. D., et al. (2016). "A Novel Application for the Detection of an Irregular Pulse Using a Smartwatch." *Heart Rhythm*.  
[2] Li, Q., et al. (2015). "A survey of ECG signal processing and analysis."  
[3] National Center for Biotechnology Information. (2015). *Making Healthcare Safer III: A Critical Analysis of Existing and Emerging Patient Safety Practices*.  
[4] Singh, O., et al. (2007). "Denoising of ECG signals using wavelet transform."  
[5] Goldberger, A. L., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals." *Circulation*.