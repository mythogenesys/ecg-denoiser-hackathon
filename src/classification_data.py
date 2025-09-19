# src/classification_data.py
import wfdb
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import Dataset
from .data_utils import load_and_resample_signal, TARGET_FS, CLEAN_ECG_DATA_PATH

# --- Configuration ---
# Define the five classes based on the AAMI standard for ECG classification
BEAT_CLASSES = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # Supraventricular (S)
    'V': 2, 'E': 2,                         # Ventricular (V)
    'F': 3,                                 # Fusion (F)
    'Q': 4, '?': 4, '|': 4, '/': 4,         # Unknown (Q)
}

# We will extract a window of this many samples around each R-peak
# 128 samples at 250 Hz is ~0.5 seconds, enough to capture the QRS and surroundings.
BEAT_WINDOW_SIZE = 128

def load_all_beats_from_dataset(data_path):
    """
    Scans the entire MIT-BIH dataset and extracts every annotated heartbeat
    into a list of (beat_signal, label) pairs.
    """
    all_beats = []
    all_labels = []
    
    record_names = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]
    
    print("Extracting all annotated heartbeats from the dataset...")
    for rec_name in tqdm(sorted(list(set(record_names)))):
        try:
            record_path = os.path.join(data_path, rec_name)
            
            # Load the resampled signal
            signal = load_and_resample_signal(record_path, TARGET_FS)
            if signal is None:
                continue
                
            # Load the annotations
            annotation = wfdb.rdann(record_path, 'atr')
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol
            # --- THIS IS THE NEW, PATCHING LINE ---
            ann_samples = ann_samples.astype('int64') # Prevent overflow before multiplication

            # Rescale annotation samples to match the new sampling frequency
            original_fs = annotation.fs
            rescaled_ann_samples = np.round(ann_samples * (TARGET_FS / original_fs)).astype(int)

            for i, symbol in enumerate(ann_symbols):
                if symbol in BEAT_CLASSES:
                    label = BEAT_CLASSES[symbol]
                    r_peak_loc = rescaled_ann_samples[i]
                    
                    # Extract the window around the R-peak
                    start = r_peak_loc - BEAT_WINDOW_SIZE // 2
                    end = r_peak_loc + BEAT_WINDOW_SIZE // 2
                    
                    # Ensure the window is within the signal bounds
                    if start >= 0 and end < len(signal):
                        beat_segment = signal[start:end]
                        all_beats.append(beat_segment)
                        all_labels.append(label)
        except Exception as e:
            print(f"Warning: Could not process record {rec_name}. Error: {e}")

    print(f"Extracted a total of {len(all_beats)} beats.")
    print(f"Label distribution: {Counter(all_labels)}")
    
    return np.array(all_beats, dtype=np.float32), np.array(all_labels, dtype=np.int64)

class ECGBeatDataset(Dataset):
    """PyTorch Dataset for ECG beat classification."""
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Add a channel dimension for the CNN
        signal_tensor = torch.from_numpy(self.signals[idx]).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal_tensor, label_tensor

# --- Verification Block ---
if __name__ == '__main__':
    beats, labels = load_all_beats_from_dataset(CLEAN_ECG_DATA_PATH)
    
    # This will create and save the processed data for faster loading next time.
    # We save it in the root project directory for simplicity.
    print("Saving processed beats and labels to .npy files...")
    np.save('all_beats.npy', beats)
    np.save('all_labels.npy', labels)
    print("✅ Saved successfully.")