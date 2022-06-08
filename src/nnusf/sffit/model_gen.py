import tensorflow as tf
from tensorflow.keras import layers

from .layers import Chi2Layer


def generate_models(units_per_layer, activation_per_layer, 
                   tr_theory_grid, tr_invcovmat, tr_pseudodata,
                   vl_theory_grid, vl_invcovmat, vl_pseudodata,
                   initializer_seed=0, output_nodes=8, output_activation="linear"):
    """
    Function that prepares the parametrization of the structure function using
    tf.keras.layers.Dense layers
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=initializer_seed)

    # (None,3) where None leaves the ndat size free such that we can use the 
    # same input layer for models with different input sizes (e.g. training and
    # validation)
    input_layer = layers.Input(
        shape=(None, 3), batch_size=1, name="input_layer"
    )  

    dense_layers = []
    for units, activation in zip(units_per_layer, activation_per_layer):
        dense_layers.append(layers.Dense(units = units, activation=activation, kernel_initializer=initializer))

    sf_output = layers.Dense(output_nodes, activation=output_activation, name="SF_output")

    # Connect all the dense layers in the model
    for dense_layer in dense_layers:
        input_layer = dense_layer(input_layer)
    sf_basis = sf_output(input_layer)

    # Add the layers that calculate the chi2 for trainig and validation
    tr_layer = Chi2Layer(tr_theory_grid, tr_invcovmat, tr_pseudodata)
    vl_layer = Chi2Layer(vl_theory_grid, vl_invcovmat, vl_pseudodata)
    tr_output = tr_layer(sf_basis)
    vl_output = vl_layer(sf_basis)

    # Initialize the model
    tr_model = tf.keras.Model(inputs=input_layer, outputs=tr_output)
    vl_model = tf.keras.Model(inputs=input_layer, outputs=vl_output)

    return tr_model, vl_model
