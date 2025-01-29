import flwr as fl
import os
import hydra
from omegaconf import DictConfig
import tenseal as ts
import pickle
from flwr.server.strategy import FedAvg


# Load the public context for homomorphic encryption
with open("public_context.pkl", "rb") as f:
    public_context = pickle.load(f)

context = ts.context_from(public_context)


class HomomorphicFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_accuracies = {}  # Track client accuracies
        self.flagged_clients = set()  # Store flagged clients

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        encrypted_params = []
        weights = []
        for client, fit_res in results:
            # Display each client's accuracy and loss
            cid = fit_res.metrics.get("cid")
            accuracy = fit_res.metrics.get("accuracy", 0.0)
            loss = fit_res.metrics.get("val_loss", 0.0)
            print(f"[Round {server_round}] Client {cid} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

            # Check if the client is flagged
            if cid in self.flagged_clients:
                print(f"Skipping flagged client {cid}")
                continue

            # Collect parameters and weights only from non-flagged clients
            client_params = []
            for param in fl.common.parameters_to_ndarrays(fit_res.parameters):
                ckks_vector = ts.lazy_ckks_vector_from(param.tobytes())
                ckks_vector.link_context(context)
                client_params.append(ckks_vector)
            encrypted_params.append(client_params)
            weights.append(fit_res.num_examples)

        # Update accuracies for client evaluation
        for _, fit_res in results:
            cid = fit_res.metrics.get("cid")
            accuracy = fit_res.metrics.get("accuracy")
            print(f"cid = {cid} accuracy={accuracy}")

            if cid:
                if cid not in self.client_accuracies:
                    self.client_accuracies[cid] = []
                self.client_accuracies[cid].append(accuracy)

                # Check if the client is malicious
                if len(self.client_accuracies[cid]) >= 3:  # Check last 3 rounds
                    recent_accuracies = self.client_accuracies[cid][-3:]
                    avg_accuracy = sum(recent_accuracies) / len(recent_accuracies)
                    if avg_accuracy < 0.5:  # Threshold for malicious client
                        self.flagged_clients.add(cid)
                        print(f"Client {cid} flagged as malicious!")
                        for i in self.flagged_clients:
                            print(f"THIS IS THE FLAGGED CLIENTS SET --- {i}")

        # Perform weighted aggregation only with non-flagged clients
        if not weights:  # Handle case where all clients are flagged
            print("No clients available for aggregation. Returning None.")
            return None, {}

        total_examples = sum(weights)
        normalized_weights = [w / total_examples for w in weights]
        aggregated_params = [param * normalized_weights[0] for param in encrypted_params[0]]
        for client_params, weight in zip(encrypted_params[1:], normalized_weights[1:]):
            for i in range(len(aggregated_params)):
                aggregated_params[i] += client_params[i] * weight

        aggregated_serialized = [param.serialize() for param in aggregated_params]
        return fl.common.ndarrays_to_parameters(aggregated_serialized), {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        # Display each client's evaluation metrics
        for client_id, fit_res in results:
            # Check if the client is flagged
            if client_id in self.flagged_clients:
                print(f"Skipping flagged client {client_id} in evaluation")
                continue

            client_accuracy = fit_res.metrics.get("accuracy", 0.0)
            client_val_loss = fit_res.metrics.get("val_loss", 0.0)
            print(f"[Round {server_round}] Client {client_id} - Evaluation Loss: {client_val_loss:.4f}, Evaluation Accuracy: {client_accuracy:.4f}")

            # Aggregate results only from non-flagged clients
            total_loss += client_val_loss * fit_res.num_examples
            total_accuracy += client_accuracy * fit_res.num_examples
            total_samples += fit_res.num_examples

        # Avoid division by zero if all clients are flagged
        if total_samples > 0:
            average_loss = total_loss / total_samples
            average_accuracy = total_accuracy / total_samples
        else:
            average_loss = 0.0
            average_accuracy = 0.0

        # Print the aggregated results
        print(f"Round {server_round} - Average loss: {average_loss:.4f}, Average accuracy: {average_accuracy:.4f}")

        return average_loss, {"accuracy": average_accuracy}



@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
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
