import flwr as fl
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="base")
def main(cfg: DictConfig):
    
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=cfg.server.min_fit_clients,
        min_eval_clients=cfg.server.min_eval_clients,
        min_available_clients=cfg.server.min_available_clients,
        fraction_fit=cfg.strategy.parameters.fraction_fit,
        fraction_eval=cfg.strategy.parameters.fraction_eval,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8000",
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy,
    )

# Entry point to run the server
if __name__ == "__main__":
    main()
