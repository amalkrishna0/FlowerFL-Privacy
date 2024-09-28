# libraries used
from typing import Dict, Tuple  # for type hinting
from flwr.common import NDArrays  # for handling numpy arrays in federated learning
import tensorflow as tf  # tensorflow for defining and training the model
import flwr as fl  # flower framework for federated learning
from model import model  # import the model architecture from model.py

# load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# class for the mnist client, inherits from flwr's NumPyClient
class MNIST(fl.client.NumPyClient):
    
    # method to retrieve the model parameters (weights) for sending to the server
    def get_parameters(self, config):
        return model.get_weights()  # returns the current model weights
    
    # method to train the model on local data with the parameters received from the server
    def fit(self, parameters, config):
        model.set_weights(parameters)  # set the received parameters as the current model's weights
        model.fit(x_train, y_train, epochs=1, batch_size=32)  # train the model on local data
        return model.get_weights(), len(x_train), {}  # return updated weights and the number of samples used
    
    # method to evaluate the model on local test data
    def evaluate(self, parameters, config):
        model.set_weights(parameters)  # set the received parameters as the current model's weights
        loss, accuracy = model.evaluate(x_test, y_test)  # evaluate the model on the test data
        return loss, len(x_test), {"accuracy": accuracy}  # return the loss, number of test samples, and accuracy

# start the mnist client and connect it to the server
fl.client.start_numpy_client(
    server_address="127.0.0.1:8000",  # connect to the server at localhost:8000
    client=MNIST()  # use the MNIST client class
)
