import flwr as fl
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import pickle
import tenseal as ts
from flwr.server.strategy import FedAvg


with open("public_context.pkl", "rb") as f:
    public_context = pickle.load(f)

context = ts.context_from(public_context)


class HomomorphicFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        encrypted_params = []
        weights = []
        for client, fit_res in results:
            client_params = []
            for param in fl.common.parameters_to_ndarrays(fit_res.parameters):
                ckks_vector = ts.lazy_ckks_vector_from(param.tobytes())
                ckks_vector.link_context(context)
                
                partition_id = fit_res.metrics["partition_id"]
                print(f"Encrypted parameter received on server (partition_id {partition_id}): {ckks_vector}")

                client_params.append(ckks_vector)
            encrypted_params.append(client_params)
            weights.append(fit_res.num_examples)

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

        total_loss = sum(r.metrics["val_loss"] * r.num_examples for _, r in results)
        total_accuracy = sum(r.metrics["accuracy"] * r.num_examples for _, r in results)
        total_samples = sum(r.num_examples for _, r in results)

        average_loss = total_loss / total_samples
        average_accuracy = total_accuracy / total_samples

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
