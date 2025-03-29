import flwr as fl
import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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

# Load the secret context for homomorphic encryption
# This ensures encrypted communication for model updates
with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)

MODEL_PATH="model_updates"
# Load trained autoencoder
autoencoder = Autoencoder(93322)
autoencoder.load_state_dict(torch.load("best_autoencoder_2.pth"))
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
            self.perturb_params(saved_params, noise_level=0.04)

            print(f"✅ Model parameters have been perturbed for {file_name}.")

        # 🔹 Step 4: Encrypt Perturbed Parameters
        encrypted_params = [ts.ckks_vector(context, param.flatten()).serialize() for param in params]
        
        return encrypted_params

    def perturb_params(self, params, noise_level):
        perturbed_params = []
        for param in params:
            param_tensor = torch.tensor(param)  # Ensure it's a tensor
            noise = torch.randn_like(param_tensor) * noise_level
            perturbed_params.append(param_tensor + noise)
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
        train(self.net, self.trainloader, epochs=5)
        val_loss, accuracy = test(self.net, self.valloader)
        print(f"Config received: {config}")  # Debugging
        latent_representation = self.extract_latent_representation(self.net)
        print("TYPE LATENT",type(latent_representation))
        np.save(f"latent_representations/client_{self.cid}.npy", latent_representation)

        return self.get_parameters(config={}), len(self.trainloader.dataset), {"partition_id": self.cid,"cid": self.cid,"accuracy":float(accuracy),"loss":float(val_loss),"latent_representation": f"latent_representations/client_{self.cid}.npy"}


    def extract_latent_representation(self,model):
        params = [param.cpu().detach().numpy().flatten() for param in model.parameters()]
        params_tensor = torch.tensor(np.concatenate(params), dtype=torch.float32)
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
        val_loss, accuracy = test(self.net, self.valloader)
        return float(val_loss), len(self.valloader.dataset), {"val_loss": float(val_loss), "accuracy": float(accuracy), "cid": self.cid}
    
    def save_model_update(self, params,path):
        """Save model parameters locally as a pickle file with a timestamp."""

        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        os.makedirs(path, exist_ok=True)  # Ensure directory exists

        file_name = f"client_{self.cid}_update_{timestamp}.pkl"
        file_path = os.path.join(path, file_name)

        try:
            with open(file_path, "wb") as f:
                pickle.dump(params, f)  # Save model parameters as pickle
            print(f"[CLIENT {self.cid}] Model update saved to {file_path}")
            return file_name
        except Exception as e:
            print(f"[CLIENT {self.cid}] ERROR: Failed to save model update - {e}")
            return None



def train(net, trainloader, epochs):
    """
    Train the local model on the given dataset.
    
    Args:
        net (torch.nn.Module): Neural network model
        trainloader (DataLoader): Training dataset loader
        epochs (int): Number of training epochs
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """
    Evaluate the model on the given dataset.
    
    Args:
        net (torch.nn.Module): Neural network model
        testloader (DataLoader): Test dataset loader
    
    Returns:
        Tuple: Validation loss and accuracy
    """
    criterion = nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return val_loss / len(testloader), accuracy


def load_data(labels, malicious=False, noise=False):
    """
    Load and preprocess the MNIST dataset for the client.
    
    Args:
        labels (List): List of labels this client should handle
        malicious (bool): Whether to modify labels maliciously
    
    Returns:
        Tuple: Trainloader, validation loader, testloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)

    train_data = [(img, target) for img, target in mnist_train if target in labels]
    test_data = [(img, target) for img, target in mnist_test if target in labels]

    if malicious:
        # Generate fake labels by shuffling or assigning random labels
        print(f"[CLIENT] Malicious mode enabled: altering labels.")
        train_data = [(img, random.randint(0, 9)) for img, _ in train_data]
        test_data = [(img, random.randint(0, 9)) for img, _ in test_data]

    elif noise:
        # Introduce random noise to labels (flipping some labels)
        print(f"[CLIENT] Noise mode enabled: introducing label noise.")
        noise_ratio = 0.2  # 20% labels will be flipped
        noisy_train_data = []
        for img, label in train_data:
            if random.random() < noise_ratio:
                new_label = random.choice([l for l in range(10) if l != label])
                noisy_train_data.append((img, new_label))
            else:
                noisy_train_data.append((img, label))
        train_data = noisy_train_data
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    valloader = DataLoader(train_data, batch_size=32)
    testloader = DataLoader(test_data, batch_size=32)

    return trainloader, valloader, testloader


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
