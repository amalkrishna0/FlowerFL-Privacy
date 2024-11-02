import flwr as fl
import numpy as np
import os
import hydra  
from omegaconf import DictConfig  

"""
Function to calculate a weighted average of metrics across clients.
The accuracy is weighted based on the number of examples from each client.
"""
def weighted_average(metrics):
    accuracies = [m["accuracy"] * num_examples for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

"""
Main function to initialize and start the federated learning server using a Hydra configuration.
The Hydra configuration provides flexibility in setting server parameters like number of clients, rounds, etc.
"""
@hydra.main(config_path="conf", config_name="base" , version_base=None)
def main(cfg: DictConfig):
    """
    Define the strategy for federated learning using FedAvg.
    - min_fit_clients: Minimum number of clients required for training.
    - min_evaluate_clients: Minimum number of clients required for evaluation.
    - min_available_clients: Minimum number of clients available in the system.
    - fraction_fit: Fraction of clients selected for training.
    - fraction_evaluate: Fraction of clients selected for evaluation.
    - evaluate_metrics_aggregation_fn: Function to aggregate metrics across clients.
    """
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=cfg.server.min_fit_clients,  
        min_evaluate_clients=cfg.server.min_eval_clients, 
        min_available_clients=cfg.server.min_available_clients,
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        evaluate_metrics_aggregation_fn=weighted_average  
    )

    """
    Start the federated learning server with the given address and strategy.
    The number of communication rounds is configured through Hydra.
    """
    fl.server.start_server(
        server_address=os.getenv("FL_SERVER_ADDRESS","0.0.0.0:5555"), 
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy
    )

"""
Main entry point of the script.
The Hydra configuration is used to initialize and start the federated learning server.
"""
if __name__ == "__main__":
    main()

