# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import random
import argparse # Import argparse for command-line arguments

# Import our own modules
from model import UNet1D
from data_utils import get_all_record_names, load_and_resample_signal, get_noise_signals, create_noisy_clean_pair, TARGET_FS

# --- Configuration ---
class Config:
    # Paths (adjust for Google Colab)
    # Assumes you mount your drive at /content/drive
    DRIVE_PATH = '/content/drive/MyDrive/ecg_denoiser_hackathon/' # Adjusted path for clarity
    CLEAN_ECG_DATA_PATH = os.path.join(DRIVE_PATH, 'data/mit-bih-arrhythmia-database-1.0.0/')
    NOISE_DATA_PATH = os.path.join(DRIVE_PATH, 'data/mit-bih-noise-stress-test-database-1.0.0/')
    MODEL_SAVE_PATH = os.path.join(DRIVE_PATH, 'models/ecg_denoiser_stpc_full.pth') # Default save path
    
    # Training Hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    NUM_EPOCHS = 50 # Increase for better performance, 50 is a good start
    NUM_WORKERS = 2
    
    # Data Generation Parameters
    SEGMENT_LENGTH_SAMPLES = 2048 
    TARGET_FS = 250 # Ensure this is consistent with your data_utils
    SEGMENT_LENGTH_S = int(SEGMENT_LENGTH_SAMPLES / TARGET_FS)
    SNR_DB_MIN = -3
    SNR_DB_MAX = 12

    # Loss function weights for STPC
    W_RECON = 1.0  # Amplitude Consistency (L1)
    W_GRAD = 0.5   # Temporal-Gradient Consistency
    W_FFT = 0.3    # Spectral-Magnitude Consistency

# --- Custom Loss Functions (STPC Components) ---
class GradientLoss(nn.Module):
    """
    Computes the L1 loss between the gradients of the prediction and the target.
    This enforces sharpness and penalizes blurry outputs.
    Corresponds to the Temporal-Gradient Consistency term in the paper.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        # Using torch.diff as it's a direct implementation of the forward-difference operator G
        pred_grad = torch.diff(prediction, dim=-1)
        target_grad = torch.diff(target, dim=-1)
        return self.loss(pred_grad, target_grad)

class FFTLoss(nn.Module):
    """
    Computes the L1 loss between the magnitudes of the FFT of the prediction and target.
    This enforces similarity in the frequency domain.
    Corresponds to the Spectral-Magnitude Consistency term in the paper.
    """
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        pred_fft = torch.fft.fft(prediction, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)
        
        # Compare magnitudes (as per the paper's phase-insensitivity principle)
        pred_fft_mag = torch.abs(pred_fft)
        target_fft_mag = torch.abs(target_fft)

        return self.loss(pred_fft_mag, target_fft_mag)


# --- PyTorch Dataset ---
class PhysioNetDataset(Dataset):
    """
    A custom PyTorch dataset that generates noisy-clean ECG pairs on the fly.
    """
    def __init__(self, config, num_samples_per_epoch):
        self.config = config
        self.num_samples = num_samples_per_epoch
        
        print("Initializing dataset: loading all clean record names...")
        self.clean_record_names = get_all_record_names(config.CLEAN_ECG_DATA_PATH)
        
        print("Loading all noise signals into memory...")
        self.noise_signals = get_noise_signals(config.NOISE_DATA_PATH, config.TARGET_FS)
        
        print("Loading all clean signals into memory for faster access...")
        self.clean_signals = []
        for name in tqdm(self.clean_record_names):
            path = os.path.join(config.CLEAN_ECG_DATA_PATH, name)
            signal = load_and_resample_signal(path, config.TARGET_FS)
            if signal is not None and len(signal) > config.SEGMENT_LENGTH_SAMPLES:
                self.clean_signals.append(signal)
        
        print(f"Dataset initialized with {len(self.clean_signals)} usable clean signals.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean_signal = random.choice(self.clean_signals)
        snr_db = random.uniform(self.config.SNR_DB_MIN, self.config.SNR_DB_MAX)
        
        noisy_segment, clean_segment = None, None
        while noisy_segment is None: # Retry if generation fails
            noisy_segment, clean_segment = create_noisy_clean_pair(
                clean_signal=clean_signal,
                noise_signals=self.noise_signals,
                segment_samples=self.config.SEGMENT_LENGTH_SAMPLES,
                snr_db=snr_db
            )

        noisy_tensor = torch.from_numpy(noisy_segment.copy()).float().unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_segment.copy()).float().unsqueeze(0)

        return noisy_tensor, clean_tensor

# --- Main Training Loop ---
def train_one_epoch(loader, model, optimizer, loss_recon, loss_grad, loss_fft, scaler, config, args):
    loop = tqdm(loader, leave=True)
    total_loss = 0.0

    for noisy, clean in loop:
        noisy = noisy.to(config.DEVICE)
        clean = clean.to(config.DEVICE)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            denoised = model(noisy)
            
            # --- CONDITIONAL LOSS CALCULATION FOR ABLATION STUDY ---
            l_recon = loss_recon(denoised, clean)
            loss = config.W_RECON * l_recon
            
            # Conditionally add gradient loss
            if args.use_gradient_loss:
                l_grad = loss_grad(denoised, clean)
                loss += config.W_GRAD * l_grad
            
            # Conditionally add FFT loss
            if args.use_fft_loss:
                l_fft = loss_fft(denoised, clean)
                loss += config.W_FFT * l_fft
            # ---------------------------------------------------------

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)

def main(args):
    config = Config()
    print(f"Using device: {config.DEVICE}")
    print(f"Gradient Loss Enabled: {args.use_gradient_loss}")
    print(f"FFT Loss Enabled: {args.use_fft_loss}")
    print(f"Model will be saved to: {args.model_save_path}")

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    model = UNet1D(in_channels=1, out_channels=1).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Loss Functions
    loss_recon = nn.L1Loss()  # Amplitude loss
    loss_grad = GradientLoss()
    loss_fft = FFTLoss()
    
    # Dataset and DataLoader
    train_dataset = PhysioNetDataset(config=config, num_samples_per_epoch=10000) 
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        shuffle=True
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        avg_loss = train_one_epoch(
            train_loader, model, optimizer, loss_recon, loss_grad, loss_fft, scaler, config, args
        )
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Average Loss: {avg_loss:.6f}")

        # Save model using the path from command-line arguments
        torch.save(model.state_dict(), args.model_save_path)
        print(f"Model saved to {args.model_save_path}")

if __name__ == '__main__':
    # --- ARGUMENT PARSING FOR EXPERIMENTAL HARNESS ---
    parser = argparse.ArgumentParser(description='Train a 1D U-Net for ECG Denoising with STPC loss.')
    
    parser.add_argument(
        '--model_save_path', 
        type=str, 
        default=Config.MODEL_SAVE_PATH, 
        help='Path to save the trained model.'
    )
    # The flags below use 'store_false', meaning they are ON by default.
    # To turn one OFF, you would run: python train.py --no-gradient-loss
    parser.add_argument(
        '--no-gradient-loss', 
        dest='use_gradient_loss', 
        action='store_false', 
        help='Disable the Temporal-Gradient Consistency loss component.'
    )
    parser.add_argument(
        '--no-fft-loss', 
        dest='use_fft_loss', 
        action='store_false', 
        help='Disable the Spectral-Magnitude Consistency loss component.'
    )
    # Set default values for the flags to True
    parser.set_defaults(use_gradient_loss=True, use_fft_loss=True)

    cli_args = parser.parse_args()
    main(cli_args)