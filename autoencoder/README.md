# Autoencoder for Anomaly Detection in Model Updates

This folder contains an autoencoder-based approach for detecting anomalies in model updates during federated learning. The autoencoder is trained on normal model updates and later used to identify potential anomalies based on reconstruction loss.

## Folder Contents

1. **`autoencoder.py`**  
   - This script trains an autoencoder using model updates stored in the `model_updates/` directory.  
   - It flattens the model parameters, trains the network, and saves the trained model.  

2. **`autoencoder.pth`**  
   - The trained autoencoder model saved after training.  
   - This file is used to detect anomalies in model updates.  

3. **`detect_anomalies.py`**  
   - This script loads model updates from a local directory, passes them through the trained autoencoder, and calculates the reconstruction error.  
   - If the error exceeds a defined threshold, it flags the update as a potential anomaly.  

## Usage

### 1. Training the Autoencoder with Model Updates
Ensure that the `model_updates/` folder contains valid `.pth` files with model update parameters. Then, run:  
```bash
python autoencoder.py
```
This will:  
- Load all model updates from `model_updates/`  
- Train the autoencoder using these updates  
- Save the trained model as `autoencoder.pth`  

### 2. Running Anomaly Detection
To check for anomalies in stored model updates, execute:  
```bash
python detect_anomalies.py
```
This script will:  
- Load the trained `autoencoder.pth`  
- Pass each model update through the autoencoder  
- Compute the reconstruction loss  
- Flag updates as anomalies if their reconstruction loss exceeds the threshold  

## Customization
- **Modify `threshold` in `detect_anomalies.py`** if you want to adjust the sensitivity of anomaly detection.  
- Ensure that the `input_dim` in `autoencoder.py` matches the actual size of your model parameters.  

## Dependencies
Install the required Python packages using:  
```bash
pip install torch numpy
```

