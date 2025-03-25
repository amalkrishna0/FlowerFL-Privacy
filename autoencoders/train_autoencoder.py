import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from model import Autoencoder

# Folder where model updates are stored
MODEL_UPDATES_DIR = "model_updates/"

def load_model_parameters():
    """
    Load model parameters from all .pkl files and flatten them into tensors.
    Returns a dataset of model parameters as flattened tensors.
    """
    param_list = []

    for file in os.listdir(MODEL_UPDATES_DIR):
        if file.endswith(".pkl"):
            file_path = os.path.join(MODEL_UPDATES_DIR, file)

            with open(file_path, "rb") as f:
                model_params = pickle.load(f)  # Load parameters from pkl

            # Flatten and concatenate all parameters into a single tensor
            flattened_params = torch.cat([torch.tensor(p).view(-1) for p in model_params])

            param_list.append(flattened_params)

    # Stack all flattened model parameters into a dataset (each row = one model update)
    dataset = torch.stack(param_list)

    return dataset

# Load dataset
dataset = load_model_parameters()
input_dim = dataset.shape[1]  # Set input_dim based on the loaded dataset
print(input_dim)
# Define Autoencoder model
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=16)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Convert dataset to DataLoader for batch processing
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        latent, reconstructed = autoencoder(batch.float())  # Forward pass
        loss = criterion(reconstructed, batch.float())  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Save the trained autoencoder
torch.save(autoencoder.state_dict(), "autoencoder.pth")
print("Autoencoder training complete. Model saved as autoencoder.pth")
