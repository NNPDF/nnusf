from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import random

import temp_data

import logging, coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=log)


################################ PREPARE DATA ################################

ndat = temp_data.ndat

# make training mask
tr_ratio = temp_data.tr_ratio
tr_indices = np.array(
    random.sample(range(ndat), int(tr_ratio * ndat)), dtype=int
)
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
input_covmat = temp_data.input_covmat
input_invcovmat = np.linalg.inv(input_covmat)
tr_invcovmat = input_invcovmat[tr_filter][:, tr_filter]
vl_invcovmat = input_invcovmat[~tr_filter][:, ~tr_filter]

# get experimental central values
input_central_values = temp_data.input_central_values
tr_central_values = input_central_values[tr_filter]
vl_central_values = input_central_values[~tr_filter]

# generate pseudodata
cholesky = np.linalg.cholesky(input_covmat)
random_samples = np.random.randn(ndat)
pseudodata = input_central_values + random_samples @ cholesky
tr_pseudodata = pseudodata[tr_filter]
vl_pseudodata = pseudodata[~tr_filter]

# define the output basis of the sf parametrization
output_basis = np.array(["fl", "f2", "f3", "flbar", "2bar", "f3bar"])
output_nodes = output_basis.size


######################## INITIALIZE SF PARAMETRIZATION #########################

# Input layer
batch_size = 1
layer_input = layers.Input(
    shape=(None, 3), batch_size=batch_size
)  # (None,3) is (ndat,3)

# Create 3 layers
layer1 = layers.Dense(10, activation="tanh")
layer2 = layers.Dense(10, activation="tanh")
layer3 = layers.Dense(10, activation="tanh")
layer4 = layers.Dense(10, activation="tanh")
layer5 = layers.Dense(10, activation="tanh")
layer6 = layers.Dense(10, activation="tanh")
layer7 = layers.Dense(output_nodes, activation="linear", name="SF_output")

# Connect the layers forming the sf parametrization model
sf_basis = layer7(layer6(layer5(layer4(layer3(layer2(layer1(layer_input)))))))


############################# ADD TR/VL CHI2 LAYERS ############################
class Chi2Layer(layers.Layer):
    def __init__(self, theory_grid, invcovmat, experimental_central_value):
        super(Chi2Layer, self).__init__()
        self.theory_grid = tf.keras.backend.constant(theory_grid)
        self.invcovmat = tf.keras.backend.constant(invcovmat)
        self.experimental_central_value = tf.keras.backend.constant(
            experimental_central_value
        )

    def call(self, input):
        predictions = tf.einsum("ijk,jk->ij", input, self.theory_grid)
        distance = predictions - self.experimental_central_value
        tmp_dot = tf.tensordot(self.invcovmat, distance[0, :], axes=1)
        chi2 = tf.tensordot(distance[0, :], tmp_dot, axes=1)
        ndat = self.theory_grid.shape[0]
        tf.print(tf.math.reduce_mean(distance))
        return chi2 / ndat


tr_layer = Chi2Layer(tr_theory_grid, tr_invcovmat, tr_pseudodata)
vl_layer = Chi2Layer(vl_theory_grid, vl_invcovmat, vl_pseudodata)

tr_output = tr_layer(sf_basis)
vl_output = vl_layer(sf_basis)

# Initialize the model
tr_model = tf.keras.Model(inputs=layer_input, outputs=tr_output)
vl_model = tf.keras.Model(inputs=layer_input, outputs=vl_output)


def custom_loss(y_true, y_pred):
    "Model prediction is the chi2"
    del y_true
    return y_pred

custom_loss = lambda y_true, y_pred : y_pred


opt = tf.keras.optimizers.Adam()

tr_model.compile(optimizer=opt, loss=custom_loss)
vl_model.compile(optimizer="Adam", loss=custom_loss)


########################### EARLY STOPPING CALLBACKS ###########################
class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, vl_model, patience_epochs, vl_kinematics_array, y):
        super().__init__()
        self.vl_model = vl_model
        self.patience_epochs = patience_epochs
        self.vl_kinematics_array = vl_kinematics_array
        self.best_epoch = None
        self.best_chi2 = None
        self.y = y
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        chi2 = self.vl_model.evaluate(
            self.vl_kinematics_array, self.y, verbose=0
        )
        if self.best_chi2 == None or chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        epochs_since_best_vl_chi2 = epoch - self.best_epoch
        if epochs_since_best_vl_chi2 > self.patience_epochs:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        log.info(f"best epoch: {self.best_epoch}")
        self.model.set_weights(self.best_weights)


################################## DO THE FIT ##################################

# tf wants us to give a y input, but we make that part of the the loss function
y = tf.zeros((batch_size, 1, output_nodes))

tr_kinematics_array = tf.expand_dims(tr_kinematics_array, axis=0)
vl_kinematics_array = tf.expand_dims(vl_kinematics_array, axis=0)

patience_epochs = int(temp_data.patience * temp_data.max_epochs)
early_stopping_callback = EarlyStopping(
    vl_model, patience_epochs, vl_kinematics_array, y
)

import ipdb; ipdb.set_trace()

tr_model.fit(
    tr_kinematics_array,
    y,
    epochs=temp_data.max_epochs,
    verbose=2,
    callbacks=[early_stopping_callback],
)


#################################### TESTS #####################################

# write the structure functions to a grid
model_sf = tf.keras.Model(
    inputs=tr_model.input, outputs=tr_model.get_layer("SF_output").output
)

tr_grid_to_be_stored = model_sf.predict(tf.constant([input_kinematics_array]))
