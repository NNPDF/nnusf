import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random

import temp_data


############################# LOAD MOCK DATA #############################

ndat = temp_data.ndat

# make training mask
tr_ratio = temp_data.tr_ratio
tr_indices = np.array(random.sample(range(ndat), int(tr_ratio*ndat)), dtype=int)
tr_filter = np.zeros(ndat, dtype=bool)
tr_filter[tr_indices] = True

# get kinematic grids for input
input_kinematics_array = temp_data.input_kinematics_array
tr_kinematics_array = input_kinematics_array[tr_filter]
vl_kinematics_array = input_kinematics_array[~tr_filter]


# get theory arrays (fk-like objects)
input_theory_grid = temp_data.input_theory_grid
tr_theory_grid = input_theory_grid[tr_filter]
vl_theory_grid = input_theory_grid[~tr_filter]


# get inverse covariance matrix
input_invcovmat = np.linalg.inv(temp_data.input_covmat)
tr_invcovmat = input_invcovmat[tr_filter][:,tr_filter]
vl_invcovmat = input_invcovmat[~tr_filter][:,~tr_filter]

# get experimental central values
input_central_values = temp_data.input_central_values
tr_central_values = input_central_values[tr_filter]
vl_central_values = input_central_values[~tr_filter]


# define the output basis of the sf parametrization
output_basis = np.array(["fl", "f2", "f3", "flbar", "2bar", "f3bar"])
output_nodes = output_basis.size






######################## INITIALIZE SF PARAMETRIZATION #########################

# Input layer
batch_size = 1
layer_input = layers.Input(shape=(None,3), batch_size=batch_size) # (None,3) is (ndat,3)

# Create 3 layers
layer1 = layers.Dense(20, activation="relu")
layer2 = layers.Dense(10, activation="relu")
layer3 = layers.Dense(output_nodes, name="SF_output")

# Connect the layers forming the sf parametrization model
sf_basis = layer3(layer2(layer1(layer_input)))





############################# ADD TR/VL CHI2 LAYERS ############################
class Observable(layers.Layer):
  def __init__(self, theory_grid, invcovmat, experimental_central_value):
    super(Observable, self).__init__()
    self.theory_grid = tf.keras.backend.constant(theory_grid)
    self.invcovmat = tf.keras.backend.constant(invcovmat)
    self.experimental_central_value = tf.keras.backend.constant(experimental_central_value)

  def call(self, input):
    predictions = tf.einsum("ijk,jk->ij", input, self.theory_grid)
    distance = predictions - self.experimental_central_value
    aa = tf.tensordot(distance[0,:],self.invcovmat,axes=1)
    chi2 = tf.tensordot(aa, distance[0,:], axes=1)
    return chi2

tr_layer = Observable(tr_theory_grid, tr_invcovmat, tr_central_values)
vl_layer = Observable(vl_theory_grid, vl_invcovmat, vl_central_values)

tr_output = tr_layer(sf_basis)
vl_output = vl_layer(sf_basis)

# Initialize the model
tr_model = tf.keras.Model(inputs=layer_input, outputs=tr_output)
vl_model = tf.keras.Model(inputs=layer_input, outputs=vl_output)

def custom_loss(y_true, y_pred):
    "Model prediction is the chi2, so we just need to minimize the sum"
    return tf.keras.backend.sum(y_pred)

tr_model.compile(optimizer="Adam", loss=custom_loss)







################################## DO THE FIT ##################################

# tf wants us to give a y input, but we make that part of the the loss function
y = tf.zeros((batch_size,ndat,output_nodes))
tr_kinematics_array = tf.expand_dims(tr_kinematics_array, axis=0)
tr_model.fit(tr_kinematics_array,y,epochs=10)







############################### STORE THE GRID##################################

# write the structure functions to a grid
model_sf = tf.keras.Model(inputs=tr_model.input, outputs=tr_model.get_layer("SF_output").output)
input_grid_to_store = tf.random.uniform(shape=(batch_size,ndat,3))
grid_to_be_stored = model_sf.predict(input_grid_to_store)
