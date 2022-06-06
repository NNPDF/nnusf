import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import temp_data

ndat = temp_data.ndat

output_basis = np.array(["fl", "f1", "f2", "flbar", "f1bar", "f2bar"])
output_nodes = output_basis.size

# Input layer
batch_size = 1
layer_input = layers.Input(shape=(None,3), batch_size=batch_size) # (None,3) is (ndat,3)

# Create 3 layers
layer1 = layers.Dense(20, activation="relu", name="layer1")
layer2 = layers.Dense(10, activation="relu", name="layer2")
layer3 = layers.Dense(output_nodes, name="layer3")

# Connect the layers forming the sf parametrization model
sf_basis = layer3(layer2(layer1(layer_input)))




# Initialize the model
model = tf.keras.Model(inputs=layer_input, outputs=sf_basis)


def custom_loss(y_true, y_pred): # pylint: disable=unused-argument
    """Default loss to be used when the model is compiled with loss = Null
    (for instance if the prediction of the model is already the loss"""
    tf.keras.backend.print_tensor(y_pred, message="y_pred = ")
    tf.keras.backend.print_tensor(y_true, message="y_true = ")
    return tf.keras.backend.sum(y_pred)

model.compile(optimizer="Adam", loss=custom_loss)


test_input = tf.random.uniform(shape=(batch_size,ndat,3))

# tf wants us to give a y input, but we make that part of the the loss function
y = tf.zeros((batch_size,ndat,output_nodes))
model.fit(test_input,y)

import ipdb; ipdb.set_trace()
