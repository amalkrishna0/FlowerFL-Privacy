import os
import pickle
import torch
import numpy as np
from model import Autoencoder   # Import the trained autoencoder model

# Path to the directory containing model update files
model_updates_dir = "model_updates/"

# Get all model update files (.pth) from the directory
model_update_files = [f for f in os.listdir(model_updates_dir) if f.endswith(".pth")]

# Initialize the autoencoder model
input_dim = 93322  # Ensure this matches the input dimension used during training
autoencoder = Autoencoder(input_dim)

# Load the trained autoencoder model weights
autoencoder.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device("cpu")))
autoencoder.eval()  # Set the model to evaluation mode

# Loop through each model update file
for file_name in model_update_files:
    file_path = os.path.join(model_updates_dir, file_name)
    
    # Load model update (expecting a list of NumPy arrays)
    with open(file_path, "rb") as f:
        model_update = pickle.load(f)

    # Convert model update to PyTorch tensors
    model_params = [torch.tensor(np.array(p)) for p in model_update]

    print(f"✅ Loaded model update '{file_name}' with {len(model_params)} layers.")

    # Flatten and concatenate all parameters to match autoencoder input format
    test_update = torch.cat([p.flatten() for p in model_params]).unsqueeze(0)
    test_update = test_update.to(torch.float32)  # Ensure float32 type

    # Use the autoencoder to reconstruct the model update
    with torch.no_grad():    # No gradient calculation needed during inference
        reconstructed = autoencoder(test_update)

    # Compute Mean Squared Error (MSE) loss as reconstruction error
    mse_loss = torch.nn.functional.mse_loss(test_update, reconstructed)

    print(f"🔍 Reconstruction Loss for '{file_name}': {mse_loss.item()}")

    # Define a threshold for anomaly detection (adjust based on training results)
    threshold = 0.0001  

    # Detect anomalies based on reconstruction loss
    if mse_loss.item() > threshold:
        print(f"⚠️ Anomaly Detected in '{file_name}'! This update significantly deviates from expected patterns.")
    else:
        print(f"✅ Model update '{file_name}' is within normal range.")

    print("=" * 50)  # Separator for better readability
