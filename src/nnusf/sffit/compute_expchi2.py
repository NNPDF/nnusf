# -*- coding: utf-8 -*-
"""Compute the experimental chi2 on the central real data"""

import json

import numpy as np
import tensorflow as tf

from .layers import ObservableLayer
from .utils import chi2


def compute_exp_chi2(datainfo, nn_layers, optimizer_parameters, **kwargs):
    del kwargs

    opt_name = optimizer_parameters.pop("optimizer", "Adam")
    optimizer = getattr(tf.keras.optimizers, opt_name)
    optimizer = optimizer(**optimizer_parameters)

    kinematics = [d.kinematics for d in datainfo.values()]
    kin_inputs = [tf.expand_dims(i, axis=0) for i in kinematics]

    # TODO: Rescale the inputs
    model_inputs = []
    datasets_obs = []
    chi2_loss = []
    exp_datasets = []
    nb_dpts_dataset = {}
    for data in datainfo.values():
        # Extract theory grid coefficients & datasets
        coefficients = data.coefficients
        nb_dpts_dataset[data.name] = data.n_data
        exp_datasets.append(data.central_values)

        # Construct the input layer as placeholders
        input_layer = tf.keras.layers.Input(shape=(None, 3), batch_size=1)
        model_inputs.append(input_layer)

        # Construct the NN layers using the pre-saved one
        sf_basis = nn_layers(input_layer)

        # Construct the full observable for a given dataset
        observable = ObservableLayer(coefficients, name=data.name)(sf_basis)
        datasets_obs.append(observable)

        # Construct the loss/chi2 function
        invcovmat = np.linalg.inv(data.covmat)
        chi2_loss.append(chi2(invcovmat))

    # Reshape the exp datasets (y_true) to (1, N)
    exp_datasets = [i.reshape(1, -1) for i in exp_datasets]

    # Construct the observable models based on real data
    obsmodel = tf.keras.Model(inputs=model_inputs, outputs=datasets_obs)
    obsmodel.compile(optimizer=optimizer, loss=chi2_loss)

    chi2_values = obsmodel.evaluate(
        kin_inputs,
        y=exp_datasets,
        batch_size=1,
        verbose=0,
        return_dict=True,
    )
    normalized_chi2s = {
        datname.removesuffix("_loss"): chi2_value
        / nb_dpts_dataset[datname.removesuffix("_loss")]
        for datname, chi2_value in chi2_values.items()
        if "_loss" in datname
    }
    # Compute the experimental Chi2 only for real data
    expreal_losses = [
        chi2_real_dataset
        for n, chi2_real_dataset in chi2_values.items()
        if (("_MATCHING" not in n) and ("_loss" in n))
    ]
    expreal_pt = [v for n, v in nb_dpts_dataset.items() if "_MATCHING" not in n]

    normalized_chi2s["total_chi2"] = chi2_values["loss"] / sum(
        nb_dpts_dataset.values()
    )
    normalized_chi2s["tot_chi2_real"] = sum(expreal_losses) / sum(expreal_pt)
    return normalized_chi2s


def add_expchi2_json(replica_dir, normalized_chi2s):
    with open(f"{replica_dir}/fitinfo.json", "r+") as fstream:
        json_file = json.load(fstream)
        json_file.update({"exp_chi2s": normalized_chi2s})
        # Sets file's current position at offset.
        fstream.seek(0)
        json.dump(json_file, fstream, sort_keys=True, indent=4)
