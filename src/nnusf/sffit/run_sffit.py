# -*- coding: utf-8 -*-
"""Executable to perform the structure function fit"""
import logging
import pathlib
import shutil

import tensorflow as tf
import yaml

from . import load_data
from .model_gen import generate_models
from .train_model import perform_fit

_logger = logging.getLogger(__name__)


def main(runcard: pathlib.Path, replica: int):
    # Create a folder for the replica
    destination = runcard.with_suffix("") / f"replica_{replica}"
    if destination.exists():
        _logger.warning(f"{destination} already exists, overwriting content.")
    destination.mkdir(parents=True, exist_ok=True)

    # copy runcard to the fit folder
    shutil.copy(runcard, destination.parent / "runcard.yml")

    # load runcard
    runcard_content = yaml.safe_load(runcard.read_text())
    experiments_dict = runcard_content["experiments"]

    # load data
    data_info = load_data.load_experimental_data(experiments_dict)

    # create pseudodata and add it to the data_info object
    load_data.add_pseudodata(data_info)

    # create a training mask and add it to the data_info object
    load_data.add_tr_filter_mask(data_info, runcard_content["trvlseed"])
    tr_model, vl_model = generate_models(data_info, **runcard_content["fit_parameters"])

    # Compile the training and validationa nd perform the fit
    perform_fit(tr_model, vl_model, data_info, **runcard_content["fit_parameters"])

    # Store the models in the relevant replica subfolders
    saved_model = tf.keras.Model(
        inputs=tr_model.inputs, outputs=tr_model.get_layer("SF_output").output
    )
    saved_model.save(destination / "model")
