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
from model import Net, Autoencoder
import torch.nn as nn
import random
import uuid

# Load the secret context for homomorphic encryption
with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)

MODEL_UPDATE_DIR = "model_updates"
os.makedirs(MODEL_UPDATE_DIR, exist_ok=True)

# Load the pre-trained autoencoder for anomaly detection
INPUT_DIMS=93322
autoencoder = Autoencoder(INPUT_DIMS)
autoencoder.load_state_dict(torch.load("autoencoder/autoencoder.pth"))
autoencoder.eval()

class HomomorphicFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, testloader, malicious=False, add_noise=False):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.malicious = malicious
        self.add_noise = add_noise
        self.noise_level = 0.2 if add_noise else 0.0

    def get_parameters(self, config):
        params = [param.cpu().detach().numpy() for param in self.net.parameters()]
        
        # Apply constant noise if enabled
        if self.add_noise:
            params = [param + self.noise_level for param in params]
            print(f"[CLIENT {self.cid}] Adding constant noise {self.noise_level} to model parameters.")
        
        self.save_model_update(params)
        
        # Check for anomalies using autoencoder
        if self.is_anomalous(params):
            print(f"[CLIENT {self.cid}] Anomaly detected in model parameters!")
        
        encrypted_params = [ts.ckks_vector(context, param.flatten()).serialize() for param in params]
        return encrypted_params
    
    def is_anomalous(self, params):
        params_tensor = torch.tensor(np.concatenate([p.flatten() for p in params]), dtype=torch.float32)
        reconstructed = autoencoder(params_tensor)
        loss = torch.nn.functional.mse_loss(reconstructed, params_tensor)
        return loss.item() > 0.0001  # Threshold can be adjusted

    def save_model_update(self, params):
        """Save model parameters locally for tracking purposes."""
        file_path = os.path.join(MODEL_UPDATE_DIR, f"client_{self.cid}_update.npy")
        
        # Ensure all parameters are NumPy arrays and flatten them
        processed_params = [np.array(p).flatten() for p in params]
        
        # Convert the list of arrays into a single array
        np.save(file_path, np.array(processed_params, dtype=object))

        print(f"[CLIENT {self.cid}] Model update saved to {file_path}")

    def set_parameters(self, parameters):
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
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=5)
        val_loss, accuracy = test(self.net, self.valloader)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "partition_id": self.cid,
            "cid": self.cid,
            "accuracy": float(accuracy),
            "loss": float(val_loss)
        }

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


def load_data(labels, malicious=False):
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

    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    valloader = DataLoader(train_data, batch_size=32)
    testloader = DataLoader(test_data, batch_size=32)

    return trainloader, valloader, testloader


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--labels", nargs='+', type=int, required=True, help="Label of data this client will handle")
    parser.add_argument("--malicious", action="store_true", help="Enable malicious client behavior")
    parser.add_argument("--noise", action="store_true", help="Enable adding constant noise to model parameters")
    args = parser.parse_args()

    trainloader, valloader, testloader = load_data(args.labels, malicious=args.malicious)
    net = Net()
    
    fl.client.start_client(
        server_address=os.getenv("FL_SERVER_ADDRESS", "localhost:8080"),
        client=HomomorphicFlowerClient(str(args.labels), net, trainloader, valloader, testloader, malicious=args.malicious, add_noise=args.noise)
    )

if __name__ == "__main__":
    main()

