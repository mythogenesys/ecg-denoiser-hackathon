# src/train_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from .model import UNet1D 
from .classifier_model import ECGClassifier
from .classification_data import ECGBeatDataset

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20 # Classification trains much faster
MODEL_SAVE_PATH = 'ecg_classifier_model.pth'

def main():
    print("--- Training ECG Beat Classifier ---")
    
    # 1. Load pre-processed data
    print("Loading processed beat data...")
    beats = np.load('all_beats.npy')
    labels = np.load('all_labels.npy')

    # 2. Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        beats, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = ECGBeatDataset(X_train, y_train)
    val_dataset = ECGBeatDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize model, optimizer, and loss function
    model = ECGClassifier(num_classes=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training loop
    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for signals, targets in train_loader:
            signals, targets = signals.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 5. Validation loop
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for signals, targets in val_loader:
                signals, targets = signals.to(DEVICE), targets.to(DEVICE)
                outputs = model(signals)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        print(classification_report(all_targets, all_preds, target_names=['N', 'S', 'V', 'F', 'Q']))

    # 6. Save the final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Classifier model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()