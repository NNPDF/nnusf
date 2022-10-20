# -*- coding: utf-8 -*-
"""Executable to perform the structure function fit."""
import logging
import pathlib
from typing import Optional

import tensorflow as tf
import yaml

from . import load_data
from .compute_expchi2 import add_expchi2_json, compute_exp_chi2
from .model_gen import generate_models
from .train_model import perform_fit
from .utils import set_global_seeds

tf.get_logger().setLevel("ERROR")

_logger = logging.getLogger(__name__)


def main(
    runcard: pathlib.Path,
    replica: int,
    destination: pathlib.Path,
):
    """Run the structure function fit.

    Parameters
    ----------
    runcard : pathlib.Path
        Path to the fit runcard
    replica : int
        replica number
    destination : pathlib.Path
        Path to the output folder
    """
    if destination.exists():
        _logger.warning(f"{destination} already exists, overwriting content.")

    replica_dir = destination / f"replica_{replica}"
    replica_dir.mkdir(parents=True, exist_ok=True)

    # Load fit run card
    runcard_content = yaml.safe_load(runcard.read_text())
    expdicts = runcard_content["experiments"]

    # Set global seeds
    set_global_seeds(global_seed=runcard_content["global_seeds"] + replica)

    # Instantiate class that loads the datasets
    _, data_info = load_data.load_experimental_data(
        experiment_list=expdicts,
        input_scaling=runcard_content.get("rescale_inputs", None),
        kincuts=runcard_content.get("kinematic_cuts", {}),
    )
    # create pseudodata and add it to the data_info object
    genrep = runcard_content.get("genrep", None)
    load_data.add_pseudodata(data_info, shift=genrep)
    # create a training mask and add it to the data_info object
    load_data.add_tr_filter_mask(data_info)

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

    # Compute the chi2 wrt central real data
    chi2s = compute_exp_chi2(
        data_info,
        fit_dict["sf_model"],
        **runcard_content["fit_parameters"],
    )
    add_expchi2_json(replica_dir, chi2s)
