from typing import Dict, Tuple
from flwr.common import NDArrays
import tensorflow as tf
import flwr as fl
from model import model
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


class MNIST(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train,y_train,epochs=1,batch_size=32)
        return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss,accuracy=model.evaluate(x_test,y_test)
        return loss,len(x_test),{"accuracy":accuracy}
    

fl.client.start_numpy_client(
    server_address="127.0.0.1:8000",
    client=MNIST()
)