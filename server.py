# libraries used
import flwr as fl  # flower framework for federated learning
import hydra  # hydra for managing configurations dynamically
from omegaconf import DictConfig  # for handling configuration objects from hydra


# main function that starts the federated learning server
@hydra.main(config_path="conf", config_name="base")
def main(cfg: DictConfig):
    # setting up the federated learning strategy using fedavg
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=cfg.server.min_fit_clients,  # minimum clients needed for training
        min_evaluate_clients=cfg.server.min_eval_clients,  # minimum clients needed for evaluation
        min_available_clients=cfg.server.min_available_clients,  # minimum clients required to start a round
        fraction_fit=cfg.strategy.parameters.fraction_fit,  # fraction of clients selected for training
        fraction_evaluate=cfg.strategy.parameters.fraction_eval,  # fraction of clients selected for evaluation
    )

    # starting the flower server with the defined configuration
    fl.server.start_server(
        server_address="0.0.0.0:8000",  # listens on all interfaces on port 8000
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),  # number of training rounds
        strategy=strategy,  # the strategy for model aggregation from clients
    )

# entry point for running the federated learning server
if __name__ == "__main__":
    main()
