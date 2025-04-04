import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Autoencoder

# Folder containing legitimate model updates
TRAIN_DATA_DIR = "model_updates/"

def load_model_parameters(folder):
    """
    Load model parameters from .pkl files, flatten them, and stack into a dataset.
    Args:
        folder (str): Folder containing .pkl model update files.
    Returns:
        Tensor: Full dataset with each row representing one model update.
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

    # Stack all model parameters into a dataset
    dataset = torch.stack(param_list)
    return dataset

# Load training dataset (only legitimate clients)
train_dataset = load_model_parameters(TRAIN_DATA_DIR)

# Set input dimension dynamically
input_dim = train_dataset.shape[1]

# Define Autoencoder model
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=1024)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Convert dataset to DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training with Early Stopping
num_epochs = 100
patience = 10  # Stop if no improvement for 10 epochs
best_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    autoencoder.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        
        latent, reconstructed = autoencoder(batch.float())  # Forward pass
        loss = criterion(reconstructed, batch.float())  # Compute loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.6f}")

    # Early Stopping Check
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(autoencoder.state_dict(), "best_autoencoder.pth")  # Save best model
        print("✅ Model improved, saved new best model!")
    else:
        counter += 1
        print(f"⚠️ No improvement for {counter}/{patience} epochs")

    if counter >= patience:
        print("⏹ Early stopping triggered. Training complete!")
        break

print("🎉 Final Model Training Complete. Best model saved as best_autoencoder.pth")
