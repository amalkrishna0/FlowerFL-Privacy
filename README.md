
# Federated Learning with Homomorphic Encryption

This repository demonstrates a Federated Learning (FL) setup enhanced with Homomorphic Encryption to ensure the privacy of the model's parameters during transmission. It employs the Flower (flwr) framework for FL orchestration and TenSEAL for encryption, configured with a Hydra-based dynamic control system. Additionally, it includes an Invalid Client Detection mechanism to enhance the robustness of the system against malicious participants.

## Key Enhancements 
### Malicious Client Simulation -

The `malicious_client.py` is not part of the main federated learning workflow but is used to simulate a security breach by extracting and saving model parameters transmitted by the server. This simulation highlights potential vulnerabilities in federated learning systems and underscores the importance of encrypting model parameters, which is implemented in this project.


### Invalid Client Detection - 
The system can detect invalid (malicious) clients that consistently provide incorrect labels or underperform. Clients with low accuracy over multiple rounds are flagged and excluded from model aggregation.


## Project Structure

- **`client.py`**: Contains the federated learning client logic. It trains the model locally, encrypts the parameters using homomorphic encryption, and communicates with the server.
- **`server.py`**: The federated learning server which aggregates encrypted parameters from various clients and updates the global model.
- **`malicious_client.py`**: A mock malicious client designed to simulate the extraction and saving of received model parameters, testing the security resilience of the FL setup.
- **`model.py`**: Defines the neural network model architecture used in the federated learning clients.
- **`utils.py`**: Includes utility functions and configurations for setting up the homomorphic encryption context using TenSEAL.
- **`conf/base.yaml`**: Hydra configuration file defining server and strategy parameters like number of rounds, minimum clients, etc.
- **`extracted_parameters/`**: Directory intended for storing parameters extracted by the malicious client.
- **`output/`**: Directory where all metrics like accuracy and loss from each epoch are stored with date and timestamp as the filename.

## How to Run

### Preliminary Step
Before starting the server and clients, generate the necessary public and secret context files:
```bash
python utils.py
```

### Run the FL Server
```bash
python server.py
```
The server is launched with configurations defined in `conf/base.yaml`.

### Run FL Clients
To start a federated learning client:
```bash
python client.py --labels <labels>
```
Replace `<labels>` with the specific dataset labels assigned to the client.

### Run an Invalid (Malicious) Client
To start a client with incorrect labels (simulating an invalid client):
```bash
python client.py --labels <labels> --malicious
```
This client will be flagged and excluded from future rounds after detection.

### Output Files
- Training and evaluation metrics are stored in the `output/` directory.
- Logs will display flagged clients and their exclusion status.

## Invalid Client Detection (Server Logic)

- The server tracks client accuracies over multiple rounds.
- If a client's average accuracy over the last three rounds is below a threshold (e.g., 0.5), it is flagged as an invalid client.
- Flagged clients are excluded from both model aggregation and evaluation.
## Key Configuration Options

- **`server.num_rounds`**: The number of rounds the federated learning process should run.
- **`server.min_fit_clients`**: The minimum number of clients required to start a training round.
- **`server.min_eval_clients`**: The minimum number of clients required for evaluation.
- **`strategy.name`**: The FL strategy to use (e.g., `FedAvg` with homomorphic encryption support).





## Demo Videos
### Parameter or weight extraction using malicious clients 
This is not the main part of the project. Parameter extraction using a malicious client is used solely to demonstrate that data leakage can occur if a malicious client is connected to our server.

[![Watch the Demo Video](https://img.youtube.com/vi/l6ROONTshtQ/0.jpg)](https://www.youtube.com/watch?v=l6ROONTshtQ)

------
### Federated Learning  Clients Server Aggregation
[![Watch the Demo Video](https://img.youtube.com/vi/UILQhUqcR_g/0.jpg)](https://www.youtube.com/watch?v=UILQhUqcR_g)