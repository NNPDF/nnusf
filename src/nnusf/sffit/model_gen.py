# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers

from nnusf.sffit.layers import GenMaskLayer
from nnusf.sffit.layers import ObservableLayer

from nnusf.sffit.utils import mask_expdata
from nnusf.sffit.utils import mask_covmat
from nnusf.sffit.utils import chi2
from nnusf.sffit.utils import generate_mask


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

    initializer = tf.keras.initializers.GlorotUniform(seed=initializer_seed)

    # (None,3) where None leaves the ndat size free such that we can use the
    # same input layer for models with different input sizes (e.g. training and
    # validation)
    input_layer = layers.Input(shape=(None, 3), batch_size=1, name="input_layer")

    # make the dense layers
    dense_layers = []
    for units, activation in zip(units_per_layer, activation_per_layer):
        dense_layers.append(
            layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer=initializer,
            )
        )

    # Connect all the dense layers in the model
    dense_nest = dense_layers[0](input_layer)
    for dense_layer in dense_layers[1:]:
        dense_nest = dense_layer(dense_nest)

    # make the output layer
    sf_output = layers.Dense(
        output_units, activation=output_activation, name="SF_output"
    )

    # output layer
    sf_basis = sf_output(dense_nest)

    # Add the layers that calculate the chi2 for trainig and validation
    tr_data, vl_data = [], []
    tr_obs, vl_obs = [], []
    tr_chi2, vl_chi2 = [], []
    for data in data_info.values():
        # Extract theory grid coefficients & datasets
        coefficients = data.coefficients
        nb_dapatoints = len(coefficients)
        exp_datasets = data.pseudodata

        # Construct the full observable for a given dataset
        observable = ObservableLayer(coefficients)(sf_basis)

        # Split the datasets into training & validation
        tr_mask, vl_mask = generate_mask(nb_dapatoints, frac=data.tr_frac)
        obs_tr = GenMaskLayer(tr_mask, name=data.name)(observable)
        obs_vl = GenMaskLayer(vl_mask, name=data.name)(observable)
        tr_obs.append(obs_tr)
        vl_obs.append(obs_vl)

        expd_tr, expd_vl = mask_expdata(exp_datasets, tr_mask, vl_mask)
        tr_data.append(expd_tr)
        vl_data.append(expd_vl)

        covm_tr, covm_vl = mask_covmat(data.covmat, tr_mask, vl_mask)
        chi2_tr = chi2(covm_tr, len(expd_tr))
        chi2_vl = chi2(covm_vl, len(expd_vl))
        tr_chi2.append(chi2_tr)
        vl_chi2.append(chi2_vl)

    # Reshape the exp datasets (y_true) to (1, N)
    tr_data = [i.reshape(1, -1) for i in tr_data]
    vl_data = [i.reshape(1, -1) for i in vl_data]

    # Initialize the models for the training & validation
    tr_model = tf.keras.Model(inputs=input_layer, outputs=tr_obs)
    vl_model = tf.keras.Model(inputs=input_layer, outputs=vl_obs)

    fit_dic = {
        "tr_model": tr_model,
        "vl_model": vl_model,
        "tr_losses": tr_chi2,
        "vl_losses": vl_chi2,
        "tr_expdat": tr_data,
        "vl_expdat": vl_data,
    }

    return fit_dic
