# src/classifier_model.py
import torch
import torch.nn as nn

class ECGClassifier(nn.Module):
    """
    A simple 1D CNN for classifying individual heartbeats.
    """
    def __init__(self, num_classes=5):
        super(ECGClassifier, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Input size: 128 -> MaxPool -> 64 -> MaxPool -> 32. So, 64 * 32
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1) 
        x = self.fc_block(x)
        return x

# --- Verification Block ---
if __name__ == '__main__':
    print("--- Verifying Classifier Architecture ---")
    test_input = torch.randn(16, 1, 128)
    model = ECGClassifier(num_classes=5)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (16, 5), "Output shape is incorrect!"
    print("✅ Verification successful.")