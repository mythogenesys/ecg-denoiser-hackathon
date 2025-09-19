import torch
import numpy as np
from .model import UNet1D

# --- Configuration ---
MODEL_PATH = 'ecg_denoiser_model.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading ---
def load_model(model_path, device):
    """Loads the pre-trained U-Net model."""
    model = UNet1D(in_channels=1, out_channels=1)
    # Load the state dictionary. Use map_location for cross-device compatibility.
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval() # Set the model to evaluation mode
    print(f"Model loaded from {model_path} and moved to {device}.")
    return model

# --- Denoising Function ---
def denoise_ecg_signal(noisy_signal_np, model):
    """
    Denoises a single ECG signal using the loaded model.

    Args:
        noisy_signal_np (np.array): A 1D NumPy array representing the noisy ECG.
        model (torch.nn.Module): The pre-trained PyTorch model.

    Returns:
        np.array: The denoised 1D NumPy array.
    """
    # 1. Convert NumPy array to a PyTorch tensor
    # Add batch (B) and channel (C) dimensions: (B, C, L)
    noisy_tensor = torch.from_numpy(noisy_signal_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    noisy_tensor = noisy_tensor.to(DEVICE)

    # 2. Perform inference
    with torch.no_grad(): # Disable gradient calculations for speed
        denoised_tensor = model(noisy_tensor)

    # 3. Convert the output tensor back to a NumPy array
    # Remove batch and channel dimensions
    denoised_signal_np = denoised_tensor.squeeze(0).squeeze(0).cpu().numpy()

    return denoised_signal_np

# --- Verification Block ---
if __name__ == '__main__':
    print("--- Verifying Inference Pipeline ---")
    
    # 1. Load the model
    denoiser_model = load_model(MODEL_PATH, DEVICE)
    
    # 2. Create a dummy noisy signal (replace with real data for better testing)
    # The length should match what the model was trained on.
    dummy_noisy_signal = np.random.randn(2048) 
    
    print(f"Input signal shape: {dummy_noisy_signal.shape}")
    
    # 3. Denoise the signal
    denoised_signal = denoise_ecg_signal(dummy_noisy_signal, denoiser_model)
    
    print(f"Output signal shape: {denoised_signal.shape}")

    assert dummy_noisy_signal.shape == denoised_signal.shape, "Shape mismatch during inference!"
    print("✅ Inference verification successful.")