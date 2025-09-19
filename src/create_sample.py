# --- START OF FILE src/create_sample.py ---
import numpy as np
import os
from data_utils import load_and_resample_signal, get_noise_signals, create_noisy_clean_pair, TARGET_FS

def generate_sample_file():
    """Generates a sample noisy ECG and saves it to a CSV file in the project root."""
    print("--- Generating Sample Noisy ECG File ---")
    
    # Define paths relative to the project root. This script should be run from the root.
    clean_record_path = "data/mit-bih-arrhythmia-database-1.0.0/101"
    noise_data_path = "data/mit-bih-noise-stress-test-database-1.0.0/"
    output_path = "sample_noisy_ecg.csv" 

    try:
        # Load prerequisite signals
        print("Loading signals...")
        clean_signal = load_and_resample_signal(clean_record_path, TARGET_FS)
        noise_signals = get_noise_signals(noise_data_path, TARGET_FS)
        
        # Create one noisy segment of 2048 samples with a noticeable amount of noise
        print("Creating noisy data pair...")
        noisy_signal_np, _ = create_noisy_clean_pair(
            clean_signal=clean_signal,
            noise_signals=noise_signals,
            segment_samples=2048,
            snr_db=3  # A good, noisy example
        )
        
        if noisy_signal_np is not None:
            # Save the noisy signal to a single-column CSV file, no header
            print(f"Saving to {output_path}...")
            np.savetxt(output_path, noisy_signal_np, delimiter=",", fmt='%.6f')
            print(f"✅ Successfully created sample file at: '{output_path}'")
        else:
            print("Error: Failed to generate noisy signal.")
            
    except Exception as e:
        print(f"An error occurred. Make sure you are running this from the project root directory.")
        print(e)

if __name__ == '__main__':
    generate_sample_file()
# --- END OF FILE src/create_sample.py ---