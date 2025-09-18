import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# Define the path to a sample record from the MIT-BIH Arrhythmia Database
# NOTE: Adjust this path to match the folder name you unzipped.
DATA_PATH = 'data/mit-bih-arrhythmia-database-1.0.0/'
RECORD_NAME = '100' # This is a common example record

def main():
    """
    Loads and visualizes a sample ECG signal to verify data access.
    """
    record_path = os.path.join(DATA_PATH, RECORD_NAME)
    
    print(f"Attempting to load record from: {record_path}")

    try:
        # Load the ECG record
        record = wfdb.rdrecord(record_path)
        
        # Load the annotations (optional, but good to check)
        annotation = wfdb.rdann(record_path, 'atr')

        print("Record loaded successfully!")
        print(f"Sampling frequency: {record.fs} Hz")
        print(f"Signal shape: {record.p_signal.shape}")
        print(f"Signal units: {record.units}")
        
        # Extract the first signal channel (e.g., MLII)
        signal = record.p_signal[:, 0]
        
        # --- Visualization ---
        # Calculate time vector for the x-axis
        # We'll plot the first 10 seconds
        seconds_to_plot = 10
        sample_points_to_plot = record.fs * seconds_to_plot
        time = np.arange(sample_points_to_plot) / record.fs

        # Create the plot
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal[:sample_points_to_plot])
        plt.title(f'First 10 Seconds of ECG Record: {RECORD_NAME}')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Amplitude ({record.units[0]})')
        plt.grid(True)
        
        # Save the plot to a file
        output_filename = 'ecg_sample_plot.png'
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
        
        # Show the plot
        plt.show()

    except Exception as e:
        print(f"Error: Could not load or process the record. Please check the file path.")
        print(e)

if __name__ == "__main__":
    main()