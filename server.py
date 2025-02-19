import flwr as fl
import os
import hydra
from omegaconf import DictConfig
import tenseal as ts
import pickle
from flwr.server.strategy import FedAvg

# Load the public homomorphic encryption context from a file
with open("public_context.pkl", "rb") as f:
    public_context = pickle.load(f)

# Initialize the encryption context from the loaded data
context = ts.context_from(public_context)

class HomomorphicFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_accuracies = {}  # Dictionary to track client accuracies across rounds
        self.flagged_clients = set()  # Set to store clients flagged as malicious

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

        for client_proxy, fit_res in results:
            cid = client_proxy.cid  # Retrieve client ID
            accuracy = fit_res.metrics.get("accuracy", 0.0)
            loss = fit_res.metrics.get("val_loss", 0.0)
            print(f"[Round {server_round}] Client {cid} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

            # Skip aggregation for flagged clients
            if cid in self.flagged_clients:
                print(f"Skipping flagged client {cid}")
                continue

            # Deserialize encrypted parameters from the client
            client_params = []
            for param in fl.common.parameters_to_ndarrays(fit_res.parameters):
                ckks_vector = ts.lazy_ckks_vector_from(param.tobytes())
                ckks_vector.link_context(context)
                client_params.append(ckks_vector)
            encrypted_params.append(client_params)
            weights.append(fit_res.num_examples)

        # Track client accuracies for malicious client detection
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            accuracy = fit_res.metrics.get("accuracy")
            print(f"cid = {cid} accuracy={accuracy}")

            if cid:
                self.client_accuracies.setdefault(cid, []).append(accuracy)

                # If the last 3 rounds' accuracy is below a threshold, flag the client as malicious
                if len(self.client_accuracies[cid]) >= 3:
                    recent_accuracies = self.client_accuracies[cid][-3:]
                    avg_accuracy = sum(recent_accuracies) / len(recent_accuracies)
                    if avg_accuracy < 0.5:
                        self.flagged_clients.add(cid)
                        print(f"Client {cid} flagged as malicious!")

        # Ensure at least one valid client remains for aggregation
        if not weights:
            print("No clients available for aggregation. Returning None.")
            return None, {}

        # Perform weighted aggregation of encrypted parameters
        total_examples = sum(weights)
        normalized_weights = [w / total_examples for w in weights]
        aggregated_params = [param * normalized_weights[0] for param in encrypted_params[0]]
        for client_params, weight in zip(encrypted_params[1:], normalized_weights[1:]):
            for i in range(len(aggregated_params)):
                aggregated_params[i] += client_params[i] * weight

        # Serialize aggregated encrypted parameters for transmission
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
