import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from model import Autoencoder  # Import the Autoencoder model
import pickle

def load_model_updates(folder_path):
    """
    Loads model updates from the specified folder, flattens, and concatenates them.

    Parameters:
    folder_path (str): The directory where model update files are stored.

    Returns:
    np.ndarray: Flattened model updates as a NumPy array.
    """

    updates = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pth"):  # Ensure we load the correct files
            with open(file_path, "rb") as f:
                params = pickle.load(f)  # Load raw NumPy parameters
            
            # Flatten and concatenate all parameters
            flat_params = np.concatenate([p.flatten() for p in params])
            updates.append(flat_params)

    return np.array(updates, dtype=np.float32) if updates else None


# Training Function
def train_autoencoder(model_updates_folder):
    """
    Trains an autoencoder using federated learning model updates.

    Parameters:
    model_updates_folder (str): The directory where model update files are stored.
    """

    # Load model updates
    print(f"Checking directory: {model_updates_folder}")
    print(f"Files found: {os.listdir(model_updates_folder)}")

    data = load_model_updates(model_updates_folder)
    
    if len(data) == 0:
        print("No model updates found!")
        return

    # Determine input dimension for the autoencoder
    input_dim = data.shape[1]   # Number of parameters in the model
    autoencoder = Autoencoder(input_dim)    # Initialize autoencoder
    criterion = nn.MSELoss()    # Loss function: Mean Squared Error
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)  # Optimizer: Adam

    # Convert data to tensor
    data = torch.tensor(data)

    # Training Loop
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()   # Reset gradients
        outputs = autoencoder(data)     # Forward pass
        loss = criterion(outputs, data)     # Compute loss
        loss.backward()     # Backpropagation
        optimizer.step()        # Update weights
        
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # Save the trained autoencoder model
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    print("Autoencoder model saved as 'autoencoder.pth'")

# Train Autoencoder using real FL model updates
train_autoencoder("model_updates")
