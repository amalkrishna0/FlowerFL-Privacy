import flwr as fl
import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import tenseal as ts
import pickle
import argparse
import seaborn as sns
import torch.nn.functional as F
from dotenv import load_dotenv
import os
import torch.nn as nn
from model import Net
from typing import List

load_dotenv()

with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)


class HomomorphicFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, testloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
    def get_parameters(self, config):
        params = [param.cpu().detach().numpy() for param in self.net.parameters()]
        encrypted_params = [ts.ckks_vector(context, param.flatten()).serialize() for param in params]
        return encrypted_params

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
        self.generate_confusion_matrix()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"partition_id": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, accuracy = test(self.net, self.valloader)
        return float(val_loss), len(self.valloader.dataset), {"val_loss": float(val_loss), "accuracy": float(accuracy)}
    def generate_confusion_matrix(self):
        all_preds = []
        all_labels = []

        self.net.eval()
        with torch.no_grad():
            for images, labels in self.valloader:
                outputs = self.net(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        
def train(net, trainloader, epochs):
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

def load_data(labels: List[int]):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)

    train_data = [(img, target) for img, target in mnist_train if target in labels]
    test_data = [(img, target) for img, target in mnist_test if target in labels]

    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    valloader = DataLoader(train_data, batch_size=32)
    testloader = DataLoader(test_data, batch_size=32)

    return trainloader, valloader, testloader


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--labels", nargs='+', type=int, required=True, help="Label of data this client will handle")
    args = parser.parse_args()

    trainloader, valloader, testloader = load_data(args.labels)
    
    net = Net()
    
    fl.client.start_client(
        server_address=os.getenv("FL_SERVER_ADDRESS", "localhost:8080"),
        client=HomomorphicFlowerClient(str(args.labels), net, trainloader, valloader, testloader)
    )

if __name__ == "__main__":
    main()
