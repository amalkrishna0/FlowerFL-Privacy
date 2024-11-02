from typing import Dict, Tuple
from flwr.common import NDArrays
import tensorflow as tf
import flwr as fl
from model import model
from dotenv import load_dotenv
import os 
import numpy as np
load_dotenv()

# Load MNIST dataset
(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()



"""
Function to filter data based on the given labels.
It returns the subset of data that matches the specified labels.
"""
def filter_data_by_label(x_data, y_data, labels: Tuple[int, int]):
    filter_idx = np.isin(y_data, labels)
    return x_data[filter_idx], y_data[filter_idx]


"""Flower client for MNIST dataset with specific labels"""
class MNISTLabelClient(fl.client.NumPyClient):
    def __init__(self, labels: Tuple[int, int]):
        """
        Initializes the client with filtered training and testing data based on the provided labels.
        Also compiles the model 
        """
        self.labels = labels
        self.x_train, self.y_train = filter_data_by_label(x_train_full, y_train_full, labels)
        self.x_test, self.y_test = filter_data_by_label(x_test_full, y_test_full, labels)
        
        print(f"Client with labels {labels} - Train size: {len(self.x_train)}, Test size: {len(self.x_test)}")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])        


    """
    Method to retrieve the current model weights to send to the server.
    """
    def get_parameters(self, config):
        return model.get_weights()



    """
    Method to train the model locally on the client's data.
    The model's weights are updated after training and sent back to the server.
    """
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=1)
        return model.get_weights(), len(self.x_train), {}



    """
    Method to evaluate the model on the client's local test data.
    After evaluation, the loss and accuracy are sent back to the server.
    """
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=1)
        print(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}") 
        return loss, len(self.x_test), {"accuracy": accuracy}


"""
Function to start the Flower client.
The client connects to the server using the specified server address and labels.
"""
def start_client(labels: Tuple[int, int]):
    fl.client.start_client(
        server_address=os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:5555"),  
        client=MNISTLabelClient(labels)
    )


"""
Main entry point of the script.
The client is started with a label provided as a command-line argument.
"""
if __name__ == "__main__":
    import sys
    label = int(sys.argv[1])
    start_client(label)