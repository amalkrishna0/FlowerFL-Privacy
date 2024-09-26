import flwr as fl
import numpy as np
import os
import time

class MaliciousClient(fl.client.NumPyClient):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_parameters(self, config):
        return None  

    def fit(self, parameters, config):
        print("Extracted Model Parameters:", parameters)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        for idx, param in enumerate(parameters):
            filename = os.path.join(self.output_dir, f"parameter_{timestamp}_{idx}.npy")
            np.save(filename, param)  
            print(f"Saved parameter {idx} to {filename}")
        
        return parameters, 0, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

output_directory = "extracted_parameters"

fl.client.start_numpy_client(
    server_address="localhost:8000", 
    client=MaliciousClient(output_dir=output_directory)
)
