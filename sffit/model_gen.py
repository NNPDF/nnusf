import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import temp_data

ndat = temp_data.ndat
input_theory_grid = temp_data.input_theory_grid

input_central_values = temp_data.input_central_values

input_invcovmat = np.linalg.inv(temp_data.input_covmat)

output_basis = np.array(["fl", "f2", "f3", "flbar", "2bar", "f3bar"])
output_nodes = output_basis.size

# Input layer
batch_size = 1
layer_input = layers.Input(shape=(ndat,3), batch_size=batch_size) # (None,3) is (ndat,3)

# Create 3 layers
layer1 = layers.Dense(20, activation="relu", name="layer1")
layer2 = layers.Dense(10, activation="relu", name="layer2")
layer3 = layers.Dense(output_nodes, name="layer3")

# Connect the layers forming the sf parametrization model
sf_basis = layer3(layer2(layer1(layer_input)))

class Observable(layers.Layer):
  def __init__(self, theory_grid, invcovmat, experimental_central_value):
    super(Observable, self).__init__()
    self.theory_grid = tf.keras.backend.constant(theory_grid)
    self.invcovmat = tf.keras.backend.constant(invcovmat)
    self.experimental_central_value = tf.keras.backend.constant(experimental_central_value)

  def call(self, input):
    predictions = tf.einsum("ijk,jk->ij", input, self.theory_grid)
    distance = predictions - self.experimental_central_value
    chi2 = tf.einsum("ij,jk,ik", distance, self.invcovmat, distance)
    return chi2

tr_layer = Observable(input_theory_grid, input_invcovmat, input_central_values)

tr_output = tr_layer(sf_basis)

# Initialize the model
model = tf.keras.Model(inputs=layer_input, outputs=tr_output)

import ipdb; ipdb.set_trace()

def custom_loss(y_true, y_pred):
    "Model prediction is the chi2, so we just need to minimize the sum"
    return tf.keras.backend.sum(y_pred)

model.compile(optimizer="Adam", loss=custom_loss)


test_input = tf.random.uniform(shape=(batch_size,ndat,3))

# tf wants us to give a y input, but we make that part of the the loss function
y = tf.zeros((batch_size,ndat,output_nodes))
model.fit(test_input,y)
