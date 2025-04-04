import flwr as fl
import os
import hydra
from omegaconf import DictConfig
import tenseal as ts
import pickle
from flwr.server.strategy import FedAvg
import numpy as np
from model import Autoencoder
import torch


import logging
import sys
class DualOutput:
    """Custom stream class to write to both stdout and a log file."""
    
    def __init__(self, file_path):
        self.terminal = sys.stdout  # Store original stdout
        self.log = open(file_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)  # Print to terminal
        self.log.write(message)  # Write to log file

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
# Define log format
log_format = "%(asctime)s - %(levelname)s - %(message)s"

# Create a file handler (writes logs to log.txt)
file_handler = logging.FileHandler("log.txt", mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(log_format))

# Create a console handler (prints logs to terminal)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(log_format))

# Get the Flower logger and add handlers
flower_logger = logging.getLogger("flwr")
flower_logger.setLevel(logging.INFO)
flower_logger.addHandler(file_handler)
flower_logger.addHandler(console_handler)

# Also add handlers to the root logger (for custom logs)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Redirect stdout and stderr to DualOutput, so both print() and logging work
sys.stdout = DualOutput("log.txt")
sys.stderr = sys.stdout



# Load the public homomorphic encryption context from a file
with open("public_context.pkl", "rb") as f:
    public_context = pickle.load(f)

# Initialize the encryption context from the loaded data
context = ts.context_from(public_context)

autoencoder = Autoencoder(93322)
autoencoder.load_state_dict(torch.load("autoencoder_new.pth", map_location=torch.device('cpu')))
autoencoder.eval()

class HomomorphicFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_accuracies = {}  # Dictionary to track client accuracies across rounds
        self.flagged_clients = set()  # Set to store clients flagged as malicious
        self.client_anomaly_counts = {}  # Track anomaly counts
        self.min_anomaly_threshold =0.5 # Initial threshold
        self.max_anomaly_threshold =0.8 # Initial threshold
        self.max_reconstruction_error = 0.8  # Dynamic thresholding variable
        
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training, excluding flagged clients."""
        available_clients = list(client_manager.all().values())  # Retrieve all clients
        filtered_clients = [client for client in available_clients if client.cid not in self.flagged_clients]
        print(f"Available clients after filtering: {[c.cid for c in filtered_clients]}")
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation, excluding flagged clients."""
        available_clients = list(client_manager.all().values())  # Retrieve all clients
        filtered_clients = [client for client in available_clients if client.cid not in self.flagged_clients]
        print(f"Available clients after filtering: {[c.cid for c in filtered_clients]}")
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model updates from clients, excluding flagged clients."""
        if not results:
            return None, {}

        encrypted_params = []  # Store encrypted model parameters from clients
        weights = []  # Store weights (number of samples per client) for aggregation
        latent_representations = {}  # Store latent representations
        anomalies = set()  # Track anomalous clients

        for client_proxy, fit_res in results:
            cid = client_proxy.cid  # Retrieve client ID
            accuracy = fit_res.metrics.get("accuracy", 0.0)
            loss = fit_res.metrics.get("loss", 0.0)

            print(f"[Round {server_round}] Client {cid} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

            if cid in self.flagged_clients:
                print(f"Skipping flagged client {cid}")
                continue

            # Load latent representation from file
            latent_path = fit_res.metrics.get("latent_representation", None)
            if latent_path and os.path.exists(latent_path):
                latent_representations[cid] = np.load(latent_path)
                print(f"Loaded latent representation from {latent_path} for Client {cid}")
            else:
                print(f"⚠️ Warning: Latent representation file not found for Client {cid}")
                latent_representations[cid] = None

            # Deserialize encrypted parameters
            client_params = []
            for param in fl.common.parameters_to_ndarrays(fit_res.parameters):
                ckks_vector = ts.lazy_ckks_vector_from(param.tobytes())
                ckks_vector.link_context(context)
                client_params.append(ckks_vector)
            encrypted_params.append(client_params)
            weights.append(fit_res.num_examples)

        # ✅ Perform anomaly detection based on reconstruction error
        for cid, latent in latent_representations.items():
            if latent is None:
                continue  # Skip if no latent representation

            latent_tensor = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
            reconstructed = autoencoder.decoder(latent_tensor)
            latent_reconstructed = autoencoder.encoder(reconstructed)
            reconstruction_error = torch.mean((latent_tensor - latent_reconstructed) ** 2).item()

            print(f"Client {cid} Reconstruction Error: {reconstruction_error:.6f}")
            if (server_round>=15):
                if reconstruction_error > self.max_anomaly_threshold or reconstruction_error < self.min_anomaly_threshold:
                    self.client_anomaly_counts[cid] = self.client_anomaly_counts.get(cid, 0) + 1
                    if self.client_anomaly_counts[cid] >= 4:  # Flag client permanently
                        self.flagged_clients.add(cid)
                        self.max_reconstruction_error = max(self.max_reconstruction_error, reconstruction_error)
                        self.anomaly_threshold = self.max_reconstruction_error  # Update threshold dynamically
                        print(f"🚨 Client {cid} flagged! New anomaly threshold: {self.anomaly_threshold:.6f}")
                    else:
                        print(f"⚠️Client {cid} reconstruction error greater than threshold : ({self.client_anomaly_counts[cid]}/{4})")
                else:
                    print(f"✅ Client {cid} passed anomaly check.")

        # ✅ Track client accuracies for malicious detection
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            accuracy = fit_res.metrics.get("accuracy")

            if cid:
                self.client_accuracies.setdefault(cid, []).append(accuracy)

                # If the last 3 rounds' accuracy is below a threshold, flag the client as malicious
                if len(self.client_accuracies[cid]) >= 3:
                    recent_accuracies = self.client_accuracies[cid][-3:]
                    avg_accuracy = sum(recent_accuracies) / len(recent_accuracies)
                    if avg_accuracy < 0.3:
                        self.flagged_clients.add(cid)
                        print(f"Client {cid} flagged as malicious!")

        # Ensure at least one valid client remains for aggregation
        if not weights:
            print("No clients available for aggregation. Returning None.")
            return None, {}

        # Perform weighted aggregation
        total_examples = sum(weights)
        normalized_weights = [w / total_examples for w in weights]
        aggregated_params = [param * normalized_weights[0] for param in encrypted_params[0]]

        for client_params, weight in zip(encrypted_params[1:], normalized_weights[1:]):
            for i in range(len(aggregated_params)):
                aggregated_params[i] += client_params[i] * weight

        # Serialize aggregated encrypted parameters
        aggregated_serialized = [param.serialize() for param in aggregated_params]

        return fl.common.ndarrays_to_parameters(aggregated_serialized), {}


    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics from clients, excluding flagged clients."""
        if not results:
            return None, {}

        total_loss, total_accuracy, total_samples = 0.0, 0.0, 0

        for client_proxy, eval_res in results:
            cid = client_proxy.cid
            client_accuracy = eval_res.metrics.get("accuracy", 0.0)
            client_val_loss = eval_res.metrics.get("val_loss", 0.0)

            # Skip evaluation results from flagged clients
            if cid in self.flagged_clients:
                print(f"Skipping flagged client {cid} in evaluation")
                continue

            print(f"[Round {server_round}] Client {cid} - Evaluation Loss: {client_val_loss:.4f}, Evaluation Accuracy: {client_accuracy:.4f}")
            total_loss += client_val_loss * eval_res.num_examples
            total_accuracy += client_accuracy * eval_res.num_examples
            total_samples += eval_res.num_examples

        # Compute average loss and accuracy, avoiding division by zero
        average_loss = total_loss / total_samples if total_samples > 0 else 0.0
        average_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0

        print(f"Round {server_round} - Average loss: {average_loss:.4f}, Average accuracy: {average_accuracy:.4f}")
        return average_loss, {"accuracy": average_accuracy}

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Initialize and start the federated learning server."""
    strategy = HomomorphicFedAvg(
        min_fit_clients=cfg.server.min_fit_clients,
        min_evaluate_clients=cfg.server.min_eval_clients,
        min_available_clients=cfg.server.min_available_clients,
        fraction_fit=0.5,
        fraction_evaluate=0.5,
    )

    server_address = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:5555")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
