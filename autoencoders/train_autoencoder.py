import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Autoencoder

# Folders for training and validation data
TRAIN_DATA_DIR = "model_updates/"  # Legitimate clients only
VALIDATION_DATA_DIR = "model_abnormal/"  # Contains both normal & bad client parameters

def load_model_parameters(folder, validation_split=None):
    """
    Load model parameters from .pkl files, flatten them, and optionally split into training/validation sets.
    Args:
        folder (str): Folder containing .pkl model update files.
        validation_split (float or None): Split ratio for validation. If None, returns full dataset.
    Returns:
        Tensor: Full dataset if no split, else (train_data, val_data) tuple.
    """
    param_list = []

    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            file_path = os.path.join(folder, file)

            with open(file_path, "rb") as f:
                model_params = pickle.load(f)  # Load parameters

            # Flatten and concatenate all parameters into a single tensor
            flattened_params = torch.cat([torch.tensor(p).view(-1) for p in model_params])

            param_list.append(flattened_params)

    # Stack all model parameters into a dataset (each row = one model update)
    dataset = torch.stack(param_list)

    if validation_split:
        # Split dataset into training and validation sets
        num_samples = dataset.shape[0]
        val_size = int(num_samples * validation_split)

        indices = torch.randperm(num_samples)  # Shuffle indices
        val_data = dataset[indices[:val_size]]  # 30% for validation
        train_data = dataset[indices[val_size:]]  # Remaining for training

        return train_data, val_data

    return dataset

# Load datasets
train_dataset = load_model_parameters(TRAIN_DATA_DIR)  # Only normal clients
train_abnormal_dataset, val_dataset = load_model_parameters(VALIDATION_DATA_DIR, validation_split=0.3)  # Mixed clients

# Merge normal & abnormal training datasets
full_train_dataset = torch.cat([train_dataset, train_abnormal_dataset])

input_dim = full_train_dataset.shape[1]  # Set input_dim dynamically

# Define Autoencoder model
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=1024)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Convert datasets to DataLoader for batch processing
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Training with Early Stopping
num_epochs = 100
patience = 10  # Stop if no improvement for 10 epochs
best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    autoencoder.train()
    total_train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        
        latent, reconstructed = autoencoder(batch.float())  # Forward pass
        loss = criterion(reconstructed, batch.float())  # Compute loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation Phase
    autoencoder.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            latent, reconstructed = autoencoder(batch.float())  
            val_loss = criterion(reconstructed, batch.float())  
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(autoencoder.state_dict(), "best_autoencoder.pth")  # Save best model
        print("✅ Model improved, saved new best model!")
    else:
        counter += 1
        print(f"⚠️ No improvement for {counter}/{patience} epochs")

    if counter >= patience:
        print("⏹ Early stopping triggered. Training complete!")
        break

print("Final Model Training Complete. Best model saved as best_autoencoder.pth")
