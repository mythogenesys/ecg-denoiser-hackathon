# src/validate_end_to_end.py
import torch
import numpy as np
import os
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Import all our custom modules
from .data_utils import load_and_resample_signal, get_noise_signals, create_noisy_clean_pair, TARGET_FS, CLEAN_ECG_DATA_PATH
from .classification_data import BEAT_CLASSES, BEAT_WINDOW_SIZE
from .model import UNet1D
from .classifier_model import ECGClassifier

# --- Configuration ---
# CHOOSE A RECORD THAT WAS NOT IN THE MAJORITY OF THE TRAINING DATA
# Records 200-234 are often used for testing as they contain more complex arrhythmias
VALIDATION_RECORD_NAME = '201' 
NOISE_SNR_DB = 0 # Add a very high level of noise to stress-test the system

DENOISER_MODEL_PATH = 'ecg_denoiser_model.pth'
CLASSIFIER_MODEL_PATH = 'ecg_classifier_model.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Main Validation Logic ---
def main():
    print("--- Starting End-to-End Validation ---")
    print(f"Test Record: {VALIDATION_RECORD_NAME}, Noise Level: {NOISE_SNR_DB} dB SNR")
    
    # 1. Load Models
    print("Loading models...")
    denoiser = UNet1D().to(DEVICE)
    denoiser.load_state_dict(torch.load(DENOISER_MODEL_PATH, map_location=DEVICE))
    denoiser.eval()

    classifier = ECGClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    classifier.eval()

    # 2. Load the unseen, clean ECG record and its true annotations
    print("Loading unseen validation data...")
    record_path = os.path.join(CLEAN_ECG_DATA_PATH, VALIDATION_RECORD_NAME)
    clean_signal = load_and_resample_signal(record_path, TARGET_FS)
    
    # Get true beat locations and labels
    import wfdb
    annotation = wfdb.rdann(record_path, 'atr')
    true_symbols = annotation.symbol
    true_samples = annotation.sample.astype('int64') * (TARGET_FS / annotation.fs)
    
    # 3. Artificially add severe noise to the entire signal
    print("Synthesizing a highly noisy signal...")
    noise_signals = get_noise_signals("data/mit-bih-noise-stress-test-database-1.0.0/", TARGET_FS)
    # We need a noise signal as long as our clean signal
    noise_type = 'muscle_artifact' # Muscle noise is very challenging
    long_noise = np.tile(noise_signals[noise_type], int(np.ceil(len(clean_signal) / len(noise_signals[noise_type]))))[:len(clean_signal)]
    
    power_clean = np.mean(clean_signal ** 2)
    power_noise = np.mean(long_noise ** 2)
    scaling_factor = np.sqrt((power_clean / (10**(NOISE_SNR_DB / 10))) / power_noise)
    noisy_signal = clean_signal + long_noise * scaling_factor
    
    # 4. Denoise the entire signal segment by segment
    print("Denoising the signal with the U-Net model...")
    denoised_signal = np.zeros_like(noisy_signal)
    for i in range(0, len(noisy_signal), 2048):
        segment = noisy_signal[i:i+2048]
        # Pad the last segment if it's too short
        if len(segment) < 2048:
            padded = np.zeros(2048)
            padded[:len(segment)] = segment
            segment = padded
        
        with torch.no_grad():
            tensor_in = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            tensor_out = denoiser(tensor_in).squeeze().cpu().numpy()
        
        denoised_signal[i:i+2048] = tensor_out[:len(noisy_signal[i:i+2048])]

    # 5. Classify beats from NOISY, DENOISED, and CLEAN signals
    print("Classifying beats from all three signal types...")
    signals_to_test = {'Noisy': noisy_signal, 'Denoised': denoised_signal, 'Clean (Ground Truth)': clean_signal}
    results = {}

    for name, sig in signals_to_test.items():
        predictions = []
        ground_truth = []
        
        for i, sym in enumerate(true_symbols):
            if sym in BEAT_CLASSES:
                loc = int(true_samples[i])
                start, end = loc - BEAT_WINDOW_SIZE//2, loc + BEAT_WINDOW_SIZE//2
                if start >= 0 and end < len(sig):
                    beat_window = sig[start:end]
                    with torch.no_grad():
                        tensor_in = torch.from_numpy(beat_window.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        pred_logit = classifier(tensor_in)
                        pred_label = torch.argmax(pred_logit, dim=1).item()
                    
                    predictions.append(pred_label)
                    ground_truth.append(BEAT_CLASSES[sym])
        
        results[name] = {'preds': predictions, 'truth': ground_truth}

    # 6. Report the results
    class_names = ['N', 'S', 'V', 'F', 'Q']
    for name, data in results.items():
        print(f"\n--- PERFORMANCE ON {name.upper()} SIGNAL ---")
        print(classification_report(
            data['truth'], 
            data['preds'], 
            target_names=class_names, 
            labels=range(len(class_names)), # <-- ADD THIS PARAMETER
            zero_division=0
        ))

        # Plot confusion matrix
        cm = confusion_matrix(data['truth'], data['preds'], labels=range(len(class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        disp.ax_.set_title(f'Confusion Matrix - {name} Signal')
        plt.savefig(f'confusion_matrix_{name.lower()}.png')
    
    print("\n✅ Validation complete. Check the classification reports and saved confusion matrix plots.")

if __name__ == '__main__':
    main()