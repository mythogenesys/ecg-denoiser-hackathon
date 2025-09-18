import wfdb
import numpy as np
import os
from scipy.signal import resample

# --- Constants ---
# We will work with a consistent sampling frequency. 250 Hz is a common choice for ML models.
TARGET_FS = 250 

# Paths to the datasets (relative to the project root)
CLEAN_ECG_DATA_PATH = 'data/mit-bih-arrhythmia-database-1.0.0/'
NOISE_DATA_PATH = 'data/mit-bih-noise-stress-test-database-1.0.0/'

def get_all_record_names(data_path):
    """Finds all unique record names in a PhysioNet directory."""
    all_files = os.listdir(data_path)
    # Record names are the base names of files (e.g., '100.dat' -> '100')
    # We use a set to get unique names
    record_names = sorted(list(set([f.split('.')[0] for f in all_files])))
    return [name for name in record_names if name.isdigit()] # Ensure we only get numeric record names

def load_and_resample_signal(record_path, target_fs):
    """Loads a signal from a PhysioNet record and resamples it to the target frequency."""
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0] # Use the first channel
        original_fs = record.fs

        if original_fs == target_fs:
            return signal
        
        # Resample the signal
        num_samples_target = int(len(signal) * target_fs / original_fs)
        resampled_signal = resample(signal, num_samples_target)
        return resampled_signal
    
    except Exception as e:
        print(f"Warning: Could not process record {record_path}. Error: {e}")
        return None

def get_noise_signals(noise_data_path, target_fs):
    """Loads all three types of noise and resamples them."""
    noise_types = {'bw': 'baseline_wander', 'em': 'electrode_motion', 'ma': 'muscle_artifact'}
    noises = {}
    for noise_code in noise_types:
        record_path = os.path.join(noise_data_path, noise_code)
        noise_signal = load_and_resample_signal(record_path, target_fs)
        if noise_signal is not None:
            noises[noise_types[noise_code]] = noise_signal
    return noises

def create_noisy_clean_pair(clean_signal, noise_signals, segment_length, target_fs, snr_db):
    """
    Creates a pair of (noisy_segment, clean_segment) for training.
    
    Args:
        clean_signal (np.array): The full, clean ECG signal.
        noise_signals (dict): A dictionary of available noise signals.
        segment_length (int): The length of the desired segment in seconds.
        target_fs (int): The sampling frequency.
        snr_db (int): The desired signal-to-noise ratio in decibels.
    
    Returns:
        (np.array, np.array): A tuple of (noisy_segment, clean_segment).
    """
    segment_samples = segment_length * target_fs

    # 1. Select a random segment from the clean ECG
    if len(clean_signal) < segment_samples:
        return None, None # Signal is too short
    
    start_index = np.random.randint(0, len(clean_signal) - segment_samples)
    clean_segment = clean_signal[start_index : start_index + segment_samples]

    # 2. Select a random noise type and a random segment from it
    noise_type = np.random.choice(list(noise_signals.keys()))
    noise_signal = noise_signals[noise_type]
    
    noise_start_index = np.random.randint(0, len(noise_signal) - segment_samples)
    noise_segment = noise_signal[noise_start_index : noise_start_index + segment_samples]

    # 3. Adjust noise power to achieve the target SNR
    power_clean = np.mean(clean_segment ** 2)
    power_noise_initial = np.mean(noise_segment ** 2)
    
    # Avoid division by zero
    if power_noise_initial == 0:
        return None, None

    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = power_clean / snr_linear
    
    scaling_factor = np.sqrt(target_noise_power / power_noise_initial)
    scaled_noise_segment = noise_segment * scaling_factor

    # 4. Create the noisy signal
    noisy_segment = clean_segment + scaled_noise_segment
    
    return noisy_segment.astype(np.float32), clean_segment.astype(np.float32)

# Example of how to use these functions (can be run for testing)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("--- Testing Data Utils ---")
    
    # 1. Load noise
    print("Loading noise signals...")
    all_noises = get_noise_signals(NOISE_DATA_PATH, TARGET_FS)
    print(f"Loaded noise types: {list(all_noises.keys())}")

    # 2. Load a clean ECG
    print("Loading a clean ECG record...")
    clean_record_names = get_all_record_names(CLEAN_ECG_DATA_PATH)
    sample_clean_signal = load_and_resample_signal(
        os.path.join(CLEAN_ECG_DATA_PATH, clean_record_names[0]), 
        TARGET_FS
    )
    print(f"Clean signal loaded with {len(sample_clean_signal)} samples.")

    # 3. Create a pair
    print("Creating a noisy-clean pair with SNR=6 dB...")
    noisy, clean = create_noisy_clean_pair(
        clean_signal=sample_clean_signal,
        noise_signals=all_noises,
        segment_length=10, # 10 seconds
        target_fs=TARGET_FS,
        snr_db=6 # A challenging but realistic SNR
    )

    if noisy is not None:
        print("Pair created successfully.")
        # 4. Plot for verification
        time = np.arange(len(clean)) / TARGET_FS
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(time, clean)
        plt.title("Original Clean ECG Segment")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, noisy)
        plt.title("Synthesized Noisy ECG Segment (SNR=6 dB)")
        plt.xlabel("Time (s)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('noisy_clean_pair_example.png')
        print("Saved verification plot to 'noisy_clean_pair_example.png'")
        plt.show()