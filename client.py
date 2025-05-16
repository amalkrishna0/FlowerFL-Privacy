import flwr as fl
import torch
from torch import optim
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader,random_split
import numpy as np
import tenseal as ts
import pickle
import argparse
import os
from model import Net
import torch.nn as nn
import random
import uuid
import time
from model import Autoencoder  # Import your autoencoder model
from collections import Counter

# Load the secret context for homomorphic encryption
# This ensures encrypted communication for model updates
with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)

MODEL_PATH="model_updates_for_3_client_not_legit"
# Load trained autoencoder
autoencoder = Autoencoder(93322)
autoencoder.load_state_dict(torch.load("autoencoder_new.pth", map_location=torch.device('cpu')))
autoencoder.eval()

class HomomorphicFlowerClient(fl.client.NumPyClient):

    def __init__(self, cid, net, trainloader, valloader, testloader, malicious=False, noise=False):
        """
        Initializes the federated learning client with encryption capabilities.
        
        Args:
            cid (str): Client ID
            net (torch.nn.Module): Neural network model
            trainloader (DataLoader): Training dataset loader
            valloader (DataLoader): Validation dataset loader
            testloader (DataLoader): Test dataset loader
            malicious (bool): Flag indicating if the client is malicious
        """
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.malicious = malicious
        self.noise = noise  # New flag for adding noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        # Move the model to the correct device
        self.net.to(self.device)


    def get_parameters(self, config):
        """
        Retrieves model parameters and encrypts them using CKKS homomorphic encryption.
        Args:
            config (dict): Configuration parameters
        Returns:
            List: Encrypted model parameters
        """
        params = [param.cpu().detach().numpy() for param in self.net.parameters()]

        # 🔹 Step 1: Save Model Update and Capture Filename
        file_name = self.save_model_update(params, MODEL_PATH)  # Get the filename

        # 🔹 Step 2: Load the Saved Model Update
        file_path = os.path.join(MODEL_PATH, file_name)
        with open(file_path, "rb") as f:
            saved_params = pickle.load(f)

        print(f"📌 Loaded {len(saved_params)} model parameters from {file_path}.")

        if self.noise:
            # 🔹 Step 3: Add Perturbation (Gaussian Noise)
            saved_params = self.perturb_params(saved_params, noise_level=0.2)

            print(f"✅ Model parameters have been perturbed for {file_name}.")

        # 🔹 Step 4: Encrypt Perturbed Parameters
        encrypted_params = [ts.ckks_vector(context, param.flatten()).serialize() for param in params]
        
        return encrypted_params


    def add_structured_noise(self, tensor, noise_factor):
        """Applies structured noise by flipping, scaling, and shifting."""
        noise = torch.randn_like(tensor) * torch.std(tensor) * noise_factor  # Gaussian noise
        flip_mask = torch.randint(0, 2, tensor.shape, dtype=torch.bool, device=tensor.device)  # Random flipping
        scale_mask = 1 + (torch.rand_like(tensor) - 0.5) * 0.2  # Small random scaling

        noisy_tensor = tensor + noise  # Add Gaussian noise
        noisy_tensor[flip_mask] *= -1  # Randomly flip signs
        noisy_tensor *= scale_mask  # Apply scaling

        return noisy_tensor
    

    def perturb_params(self, params, noise_level):
        """Applies structured noise by flipping, scaling, and shifting."""
        perturbed_params = []
        for param in params:
            param_tensor = torch.tensor(param)  # Ensure it's a tensor
            noisy_param = self.add_structured_noise(param_tensor, noise_level)
            perturbed_params.append(noisy_param)
        return perturbed_params

    

    def set_parameters(self, parameters):
        """
        Decrypts and sets the received model parameters.
        
        Args:
            parameters (List): Encrypted model parameters received from the server
        """
        params = []
        for param in parameters:
            serialized_param = param.tobytes()
            ckks_vector = ts.lazy_ckks_vector_from(serialized_param)
            ckks_vector.link_context(context)
            decrypted_param = ckks_vector.decrypt()
            decrypted_param = np.array(decrypted_param)
            params.append(decrypted_param)

        params_dict = zip(self.net.state_dict().keys(), params)
        state_dict = {k: torch.Tensor(v.reshape(self.net.state_dict()[k].shape)) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)




    def fit(self, parameters, config):
        """
        Train the local model with the provided parameters and return updated parameters.
        
        Args:
            parameters (List): Model parameters received from the server
            config (dict): Training configuration parameters
        
        Returns:
            Tuple: Updated model parameters, number of samples used, and client metadata
        """
        self.set_parameters(parameters)
        train(self.net, self.trainloader, self.valloader ,  epochs=10)
        val_loss, accuracy, precision, recall = test(self.net, self.valloader)
        print(f"Config received: {config}")  # Debugging
        if self.noise:
            latent_representation = self.extract_latent_representation(self.net, apply_noise=True, noise_factor=0.2)
        else:
            latent_representation = self.extract_latent_representation(self.net, apply_noise=False, noise_factor=0.2)
        print("TYPE LATENT",type(latent_representation))
        np.save(f"latent_representations/client_{self.cid}.npy", latent_representation)

        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "partition_id": self.cid,
            "cid": self.cid,
            "accuracy": float(accuracy),
            "loss": float(val_loss),
            "latent_representation": f"latent_representations/client_{self.cid}.npy"}

    def extract_latent_representation(self, model, apply_noise=True, noise_factor=0.2):
        """
        Extracts the latent representation of model parameters.
        Applies structured noise before encoding if apply_noise is True.
        """
        params = [param.cpu().detach().numpy().flatten() for param in model.parameters()]
        params_tensor = torch.tensor(np.concatenate(params), dtype=torch.float32, device=self.device)

        # ✅ Apply structured noise before encoding (to match first case)
        if apply_noise:
            params_tensor = self.add_structured_noise(params_tensor, noise_factor)

        with torch.no_grad():
            latent_representation = autoencoder.encoder(params_tensor)

        return latent_representation.numpy()


    def evaluate(self, parameters, config):
        """
        Evaluate the local model on validation data and return metrics.
        
        Args:
            parameters (List): Model parameters received from the server
            config (dict): Configuration parameters
        
        Returns:
            Tuple: Validation loss, number of samples used, and accuracy metrics
        """
        self.set_parameters(parameters)
        val_loss, accuracy, precision, recall = test(self.net, self.valloader)
        return float(val_loss), len(self.valloader.dataset), {
            "val_loss": float(val_loss),
            "val_accuracy": float(accuracy),
            "cid": self.cid
        }
    
    def save_model_update(self, params,path):
        """Save model parameters locally as a pickle file with a timestamp."""

        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        os.makedirs(path, exist_ok=True)  # Ensure directory exists

        file_name = f"client_update_{timestamp}.pkl"
        file_path = os.path.join(path, file_name)

        try:
            with open(file_path, "wb") as f:
                pickle.dump(params, f)  # Save model parameters as pickle
            print(f"[CLIENT {self.cid}] Model update saved to {file_path}")
            return file_name
        except Exception as e:
            print(f"[CLIENT {self.cid}] ERROR: Failed to save model update - {e}")
            return None



def train(net, trainloader, valloader, epochs):
    """
    Train the local model on the given dataset.
    
    Args:
        net (torch.nn.Module): Neural network model
        trainloader (DataLoader): Training dataset loader
        epochs (int): Number of training epochs
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    early_stopping_patience = 5  # Stop training if validation loss does not improve for 5 epochs
    best_val_loss = float('inf')
    early_stop_count = 0

    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation phase
        val_loss, accuracy, precision, recall = test(net, valloader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader)}, Val Loss: {val_loss}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stopping_patience:
                print("Early stopping triggered.")
                break


def test(net, testloader):
    """
    Evaluate the model on the given dataset.
    
    Returns:
        Tuple: Validation loss, accuracy, precision, recall, f1
    """
    criterion = nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return val_loss / len(testloader), accuracy, precision, recall


def load_data(labels, malicious=False, noise=False):
    """
    Load and preprocess the MNIST dataset for the client.
    
    Args:
        labels (List): List of labels this client should handle
        malicious (bool): Whether to modify labels maliciously
    
    Returns:
        Tuple: Trainloader, validation loader, testloader
    """
    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST dataset
    full_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST('data', train=False, download=True, transform=transform)

    # Filter dataset based on provided labels
    train_data = [(img, label) for img, label in full_train if label in labels]
    test_data = [(img, label) for img, label in full_test if label in labels]

    print(f"Filtered Training Images per Label: {dict(Counter([label for _, label in train_data]))}")
    print(f"Filtered Test Images per Label: {dict(Counter([label for _, label in test_data]))}")

    # Apply malicious client behavior
    if malicious:
        print("⚠️ Malicious client detected: Flipping labels randomly.")
        train_data = [(img, random.choice(range(10))) for img, _ in train_data]  # Random labels

    # Apply noise to labels (10% chance of incorrect labels)
    if noise:
        print("⚠️ Noisy client detected: Adding label noise (10%).")
        train_data = [(img, random.choice(range(10))) if random.random() < 0.5 else (img, label) for img, label in train_data]

    # Split train into train & validation
    val_size = int(len(train_data) * 0.2)
    train_data, val_data = train_data[:-val_size], train_data[-val_size:]

    print(f"Dataset Sizes - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader    
    
def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--labels", nargs='+', type=int, required=True, help="Label of data this client will handle")
    parser.add_argument("--malicious", action="store_true", help="Enable malicious client behavior")
    parser.add_argument("--noise", action="store_true", help="Enable noise in model parameters")  # New argument

    args = parser.parse_args()

    trainloader, valloader, testloader = load_data(args.labels, malicious=args.malicious, noise=args.noise)
    net = Net()

    fl.client.start_client(
        server_address=os.getenv("FL_SERVER_ADDRESS", "localhost:8080"),
        client=HomomorphicFlowerClient(str(args.labels), net, trainloader, valloader, testloader, malicious=args.malicious, noise=args.noise)
    )


if __name__ == "__main__":
    main()
