import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from layers import Chi2Layer


def generate_models(data_info, units_per_layer, activation_per_layer, 
                   initializer_seed=0, output_units=6, output_activation="linear",
                   **kwargs):
    """
    Function that prepares the parametrization of the structure function using
    tf.keras.layers.Dense layers
    """
    del kwargs # we don't use the kwargs

    initializer = tf.keras.initializers.GlorotUniform(seed=initializer_seed)

    # (None,3) where None leaves the ndat size free such that we can use the 
    # same input layer for models with different input sizes (e.g. training and
    # validation)
    input_layer = layers.Input(
        shape=(None, 3), batch_size=1, name="input_layer"
    )  

    # make the dense layers
    dense_layers = []
    for units, activation in zip(units_per_layer, activation_per_layer):
        dense_layers.append(layers.Dense(units = units, activation=activation, kernel_initializer=initializer))

    # Connect all the dense layers in the model
    for enum, dense_layer in enumerate(dense_layers):
        if enum==0:
            dense_nest = dense_layer(input_layer)
        else:
            dense_nest = dense_layer(dense_nest)

    # make the output layer
    sf_output = layers.Dense(output_units, activation=output_activation, name="SF_output")

    # output layer
    sf_basis = sf_output(dense_nest)

    # Add the layers that calculate the chi2 for trainig and validation
    tr_outputs = []
    vl_outputs = []
    tr_ndata_index_of_experiment = 0
    vl_ndata_index_of_experiment = 0
    for data in data_info.values():
        coefficients = data.coefficients
        tr_coefficients = coefficients[data.tr_filter]
        vl_coefficients = coefficients[~data.tr_filter]
        invcovmat = np.linalg.inv(data.covmat)
        tr_invcovmat = invcovmat[data.tr_filter][:, data.tr_filter]
        vl_invcovmat = invcovmat[~data.tr_filter][:, ~data.tr_filter]
        tr_pseudodata = data.pseudodata[data.tr_filter]
        vl_pseudodata = data.pseudodata[~data.tr_filter]
        tr_data_domain = [tr_ndata_index_of_experiment, tr_ndata_index_of_experiment+tr_pseudodata.size]
        vl_data_domain = [vl_ndata_index_of_experiment, vl_ndata_index_of_experiment+vl_pseudodata.size]
        tr_layer = Chi2Layer(tr_coefficients, tr_invcovmat, tr_pseudodata, tr_data_domain, training_data=True, name=data.data_name)
        vl_layer = Chi2Layer(vl_coefficients, vl_invcovmat, vl_pseudodata, vl_data_domain, training_data=False, name=data.data_name)
        tr_outputs.append(tr_layer(sf_basis))
        vl_outputs.append(vl_layer(sf_basis))
        tr_ndata_index_of_experiment += tr_pseudodata.size
        vl_ndata_index_of_experiment += vl_pseudodata.size

    tr_output = tf.stack(tr_outputs)
    vl_output = tf.stack(vl_outputs)

    # Initialize the model
    tr_model = tf.keras.Model(inputs=input_layer, outputs=tr_output)
    vl_model = tf.keras.Model(inputs=input_layer, outputs=vl_output)

    return tr_model, vl_model
