from typing import Dict, Tuple
from flwr.common import NDArrays
import tensorflow as tf
import flwr as fl
from model import model
from dotenv import load_dotenv
import os 
import numpy as np
import tenseal as ts

load_dotenv()

(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

# Helper function to filter dataset by label
def filter_data_by_label(x_data, y_data, labels: Tuple[int, int]):
    filter_idx = np.isin(y_data, labels)
    return x_data[filter_idx], y_data[filter_idx]

# Federated learning client class with encryption

class MNISTLabelClient(fl.client.NumPyClient):
    def __init__(self, labels: Tuple[int, int]):
        self.labels = labels
        self.x_train, self.y_train = filter_data_by_label(x_train_full, y_train_full, labels)
        self.x_test, self.y_test = filter_data_by_label(x_test_full, y_test_full, labels)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Set up TenSEAL encryption context for CKKS
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()

    def encrypt_parameters(self, parameters):
        # Encrypt model weights using TenSEAL (CKKS encryption)
        # Convert weights to float64 for CKKS compatibility and encrypt layer-wise
        encrypted_parameters = [ts.ckks_tensor(self.context, param.astype(np.float64)) for param in parameters]
        return encrypted_parameters

    def decrypt_parameters(self, encrypted_parameters):
        # Decrypt model weights using TenSEAL and convert back to float32
        decrypted_parameters = [param.decrypt().astype(np.float32) for param in encrypted_parameters]
        return decrypted_parameters

    def get_parameters(self, config):
        # Get the unencrypted model weights
        parameters = model.get_weights()
        # Encrypt the parameters before sending to server
        encrypted_parameters = self.encrypt_parameters(parameters)
        return encrypted_parameters

    def fit(self, parameters, config):
        # Decrypt the received parameters from the server
        decrypted_parameters = self.decrypt_parameters(parameters)
        # Set the decrypted parameters as the model's weights
        model.set_weights(decrypted_parameters)
        # Train the model
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=1)
        # Get the updated model weights and encrypt them
        updated_parameters = model.get_weights()
        encrypted_parameters = self.encrypt_parameters(updated_parameters)
        return encrypted_parameters, len(self.x_train), {}

    def evaluate(self, parameters, config):
        # Decrypt the received parameters from the server
        decrypted_parameters = self.decrypt_parameters(parameters)
        # Set the decrypted parameters as the model's weights
        model.set_weights(decrypted_parameters)
        # Evaluate the model
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=1)
        return loss, len(self.x_test), {"accuracy": accuracy}

def start_client(labels: Tuple[int, int]):
    fl.client.start_client(
        server_address=os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:5555"),
        client=MNISTLabelClient(labels)
    )
