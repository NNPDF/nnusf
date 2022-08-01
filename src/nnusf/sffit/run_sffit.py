# -*- coding: utf-8 -*-
"""Executable to perform the structure function fit."""
import logging
import pathlib
from typing import Optional

import tensorflow as tf
import yaml

from . import load_data
from .model_gen import generate_models
from .train_model import perform_fit
from .utils import set_global_seeds

tf.get_logger().setLevel("ERROR")

_logger = logging.getLogger(__name__)


def main(
    runcard: pathlib.Path,
    replica: int,
    destination: Optional[pathlib.Path],
):
    """Run the structure function fit.

    Parameters
    ----------
    runcard : pathlib.Path
        Path to the fit runcard
    replica : int
        replica number
    destination : Optional[pathlib.Path]
        Path to the output folder
    """
    if destination is None:
        destination = pathlib.Path.cwd().absolute() / runcard.stem

    if destination.exists():
        _logger.warning(f"{destination} already exists, overwriting content.")

    replica_dir = destination / f"replica_{replica}"
    replica_dir.mkdir(parents=True, exist_ok=True)

    # Load fit run card
    runcard_content = yaml.safe_load(runcard.read_text())
    experiments_dict = runcard_content["experiments"]

    # Set global seeds
    set_global_seeds(global_seed=runcard_content["global_seeds"] + replica)

    # Instantiate class that loads the datasets
    w2min = runcard_content.get("W2min", None)
    data_info = load_data.load_experimental_data(experiments_dict, w2min)
    # create pseudodata and add it to the data_info object
    genrep = runcard_content.get("genrep", None)
    load_data.add_pseudodata(data_info, shift=genrep)
    # create a training mask and add it to the data_info object
    load_data.add_tr_filter_mask(data_info, runcard_content["trvlseed"])

    # Rescale input kinematics if required
    if runcard_content.get("rescale_inputs", None):
        _logger.info("Kinematic inputs are being rescaled")
        kls, esk = load_data.cumulative_rescaling(data_info)
        load_data.rescale_inputs(data_info, kls, esk)
        runcard_content["scaling"] = {"map_from": kls, "map_to": esk}

    # Save a copy of the fit runcard to the fit folder
    with open(replica_dir.parent / "runcard.yml", "w") as fstream:
        yaml.dump(runcard_content, fstream, sort_keys=False)

    fit_dict = generate_models(data_info, **runcard_content["fit_parameters"])

    # Compile the training and validationa nd perform the fit
    log_freq = runcard_content.get("log_freq", 100)
    perform_fit(
        fit_dict,
        data_info,
        replica_dir,
        log_freq,
        **runcard_content["fit_parameters"],
    )

    # Store the models in the relevant replica subfolders
    final_placeholder = tf.keras.layers.Input(shape=(None, 3))
    saved_model = tf.keras.Model(
        inputs=final_placeholder,
        outputs=fit_dict["sf_model"](final_placeholder),
    )
    saved_model.save(replica_dir / "model")
