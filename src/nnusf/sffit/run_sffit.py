# -*- coding: utf-8 -*-
"""Executable to perform the structure function fit"""
import logging
import pathlib
import shutil
from typing import Optional

import tensorflow as tf
import yaml

from . import load_data
from .model_gen import generate_models
from .train_model import perform_fit

tf.get_logger().setLevel("WARNING")

_logger = logging.getLogger(__name__)


def main(
    runcard: pathlib.Path,
    replica: int,
    destination: Optional[pathlib.Path] = None,
):
    # Create a folder for the replica
    if destination is None:
        destination = runcard.parent / "fits" / runcard.stem
        if destination.exists():
            _logger.warning(
                f"{destination} already exists, overwriting content."
            )

    replica_dir = destination / f"replica_{replica}"
    replica_dir.mkdir(parents=True, exist_ok=True)

    # copy runcard to the fit folder
    shutil.copy(runcard, replica_dir.parent / "runcard.yml")

    # load runcard
    runcard_content = yaml.safe_load(runcard.read_text())
    experiments_dict = runcard_content["experiments"]

    # load data
    data_info = load_data.load_experimental_data(experiments_dict)

    # create pseudodata and add it to the data_info object
    load_data.add_pseudodata(data_info)

    fit_dict = generate_models(data_info, **runcard_content["fit_parameters"])

    # Compile the training and validationa nd perform the fit
    best_model = perform_fit(
        fit_dict, data_info, **runcard_content["fit_parameters"]
    )

    # Store the models in the relevant replica subfolders
    saved_model = tf.keras.Model(
        inputs=best_model.model.get_layer("dense").input,
        outputs=best_model.model.get_layer("SF_output").output,
    )
    model_path = replica_dir / "model"
    saved_model.save(model_path)
    _logger.info(f"Saved the tensorflow model to {model_path.absolute()}")
