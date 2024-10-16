import flwr as fl
import numpy as np
import os
import hydra  
from omegaconf import DictConfig  

def weighted_average(metrics):
    accuracies = [m["accuracy"] * num_examples for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

@hydra.main(config_path="conf", config_name="base" , version_base=None)
def main(cfg: DictConfig):
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=cfg.server.min_fit_clients,  
        min_evaluate_clients=cfg.server.min_eval_clients, 
        min_available_clients=cfg.server.min_available_clients,
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        evaluate_metrics_aggregation_fn=weighted_average  
    )

    fl.server.start_server(
        server_address=os.getenv("FL_SERVER_ADDRESS","0.0.0.0:5555"), 
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
