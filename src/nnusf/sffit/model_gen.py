# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .layers import GenMaskLayer, ObservableLayer
from .utils import chi2, mask_covmat, mask_expdata


def generate_models(
    data_info,
    units_per_layer,
    activation_per_layer,
    initializer_seed=0,
    output_units=6,
    output_activation="linear",
    **kwargs
):
    """
    Function that prepares the parametrization of the structure function using
    tf.keras.layers.Dense layers
    """
    del kwargs  # we don't use the kwargs

    # make the dense layers
    dense_layers = []
    for i, (units, activation) in enumerate(
        zip(units_per_layer, activation_per_layer)
    ):
        initializer = tf.keras.initializers.GlorotUniform(
            seed=initializer_seed + i
        )
        dense_layers.append(
            tf.keras.layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer=initializer,
            )
        )

    # make the output layer
    sf_output = tf.keras.layers.Dense(
        output_units, activation=output_activation, name="SF_output"
    )

    # Connect all the HIDDEN dense layers in the model
    def sequential(layer_input):
        dense_nest = dense_layers[0](layer_input)
        for dense_layer in dense_layers[1:]:
            dense_nest = dense_layer(dense_nest)
        dense_nest = sf_output(dense_nest)
        return dense_nest

    model_inputs = []
    tr_data, vl_data = [], []
    tr_obs, vl_obs = [], []
    tr_chi2, vl_chi2 = [], []
    tr_dpts, vl_dpts = {}, {}
    for data in data_info.values():
        # Extract theory grid coefficients & datasets
        coefficients = data.coefficients
        nb_dapatoints = len(coefficients)
        exp_datasets = data.pseudodata

        # Construct the input layer as placeholders
        input_layer = tf.keras.layers.Input(shape=(None, 3), batch_size=1)
        model_inputs.append(input_layer)

        # Construct the full observable for a given dataset
        sf_basis = sequential(input_layer)
        observable = ObservableLayer(coefficients)(sf_basis)

        # Split the datasets into training & validation
        tr_mask, vl_mask = data.tr_filter, ~data.tr_filter
        obs_tr = GenMaskLayer(tr_mask, name=data.name)(observable)
        obs_vl = GenMaskLayer(vl_mask, name=data.name)(observable)
        tr_obs.append(obs_tr)
        vl_obs.append(obs_vl)

        expd_tr, expd_vl = mask_expdata(exp_datasets, tr_mask, vl_mask)
        tr_data.append(expd_tr)
        vl_data.append(expd_vl)

        # Mask the covmat first before computing the inverse
        invcovmat = np.linalg.inv(data.covmat)
        invcov_tr, invcov_vl = mask_covmat(invcovmat, tr_mask, vl_mask)
        chi2_tr = chi2(invcov_tr)
        chi2_vl = chi2(invcov_vl)
        tr_chi2.append(chi2_tr)
        vl_chi2.append(chi2_vl)

        # Save the nb of datapoints for both tr&vl for later use
        tr_dpts[data.name] = len(expd_tr)
        vl_dpts[data.name] = len(expd_vl)

    # Reshape the exp datasets (y_true) to (1, N)
    tr_data = [i.reshape(1, -1) for i in tr_data]
    vl_data = [i.reshape(1, -1) for i in vl_data]

    # Initialize the models for the training & validation
    tr_model = tf.keras.Model(inputs=model_inputs, outputs=tr_obs)
    vl_model = tf.keras.Model(inputs=model_inputs, outputs=vl_obs)

    fit_dic = {
        "tr_model": tr_model,
        "vl_model": vl_model,
        "tr_losses": tr_chi2,
        "vl_losses": vl_chi2,
        "tr_expdat": tr_data,
        "vl_expdat": vl_data,
        "tr_datpts": tr_dpts,
        "vl_datpts": vl_dpts,
        "pdf_model": sequential,
    }

    return fit_dic
