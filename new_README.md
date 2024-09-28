
# Project Progress So Far
We are working on implementing federated learning with privacy protection using encryption. Before introducing the encryption mechanism, we need to demonstrate that, in a normal federated system, there is a risk of data leakage and privacy breakdown. Otherwise, there would be no point in adding encryption.

So far, we have set up a server and a set of clients (4 clients in this case), and we have also run a malicious_client script. This script connects to the server address, and while the clients are sending model updates to the server, the malicious client extracts the parameters and saves them locally in a directory in the form of NumPy files.

Up to this point, we have successfully extracted the model parameters. Moving forward, we need to demonstrate data inference using these parameters because the parameters alone are not necessarily useful to an attacker. We aim to extract data or demonstrate some form of data leakage, such as through data inference.

## Project Structure

- **`server.py`**: Script to launch the server for federated learning, which coordinates model updates from clients.
- **`client.py`**: Script for standard FL clients that train locally and send model updates to the server.
- **`malicious_client.py`**: A client designed to extract and save model parameters during training to test the system's security.
- **`model.py`**: Defines the neural network model used by both clients.
- **`conf/`**: Contains the Hydra configuration files for the server and strategy.
  - **`server_config.yaml`**: Main server configuration, including the number of clients and rounds.
  - **`strategy/`**: Configurations for different federated learning strategies (e.g., FedAvg).
- **`extracted_parameters/`**: Directory where extracted model parameters are stored by the malicious client.

## How to Run

### 1. Install Dependencies

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the FL Server

To start the federated learning server:

```bash
python server.py
```

This will launch the server with the configuration specified in `conf/server_config.yaml`.

### 3. Run FL Clients

To start a federated learning client and connect it to the server:

```bash
python client.py
```

You can simulate multiple clients by running this command in multiple terminals.

### 4. Run the Malicious Client

To run the malicious client, which extracts and stores model parameters:

```bash
python malicious_client.py
```

The extracted parameters will be saved in the `extracted_parameters/` directory.

### 5. Customize Configuration

You can customize the configuration by editing `conf/server_config.yaml` or by passing configuration overrides directly from the command line. For example, to run the server with custom parameters:

```bash
python server.py server.num_rounds=5 server.min_fit_clients=4
```

## Key Configuration Options

- **`server.num_rounds`**: Number of federated learning rounds.
- **`server.min_fit_clients`**: Minimum number of clients required to start a training round.
- **`server.min_eval_clients`**: Minimum number of clients required to perform evaluation.
- **`strategy`**: Specifies the federated learning strategy (e.g., `FedAvg`).

---

