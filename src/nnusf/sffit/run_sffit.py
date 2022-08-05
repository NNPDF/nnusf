# -*- coding: utf-8 -*-
"""Executable to perform the structure function fit."""
import copy
import json
import logging
import pathlib
from textwrap import indent
from typing import Optional

import numpy as np
import tensorflow as tf
import yaml

from . import load_data
from .compute_expchi2 import compute_exp_chi2
from .hyperscan import (
    construct_hyperfunc,
    construct_hyperspace,
    perform_hyperscan,
)
from .model_gen import generate_models
from .train_model import perform_fit
from .utils import set_global_seeds

tf.get_logger().setLevel("ERROR")

_logger = logging.getLogger(__name__)


def main(
    runcard: pathlib.Path,
    replica: int,
    nbtrials: Optional[int] = None,
    destination: Optional[pathlib.Path] = None,
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
    global_seed = runcard_content["global_seeds"] + replica
    set_global_seeds(global_seed=global_seed)

    # Instantiate class that loads the datasets
    w2min = runcard_content.get("W2min", None)
    data_info = load_data.load_experimental_data(experiments_dict, w2min)
    make_copy_raw_dataset = copy.deepcopy(data_info)
    # create pseudodata and add it to the data_info object
    genrep = runcard_content.get("genrep", None)
    load_data.add_pseudodata(data_info, shift=genrep)
    # create a training mask and add it to the data_info object
    load_data.add_tr_filter_mask(data_info)

    # Rescale input kinematics if required
    if runcard_content.get("rescale_inputs", None):
        _logger.info("Kinematic inputs are being rescaled")
        kls, esk = load_data.cumulative_rescaling(data_info)
        load_data.rescale_inputs(data_info, kls, esk)
        runcard_content["scaling"] = {"map_from": kls, "map_to": esk}

    # Save a copy of the fit runcard to the fit folder
    with open(replica_dir.parent / "runcard.yml", "w") as fstream:
        yaml.dump(runcard_content, fstream, sort_keys=False)

    # Control the hyperparameter optimisation
    if nbtrials and runcard_content["hyperscan"]:
        log_freq = runcard_content.get("log_freq", 1e10)
        hyperspace = construct_hyperspace(**runcard_content)

        def fn_hyper_train(hyperspace_dict):
            return construct_hyperfunc(
                data_info, hyperspace_dict, replica_dir, log_freq
            )

        perform_hyperscan(fn_hyper_train, hyperspace, nbtrials, replica_dir)
        return

    fit_dict = generate_models(data_info, **runcard_content["fit_parameters"])

    # Compile the training and validationa nd perform the fit
    log_freq = runcard_content.get("log_freq", 100)
    _ = perform_fit(
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
    load_data.add_pseudodata(make_copy_raw_dataset, shift=False)
    if runcard_content.get("rescale_inputs", None):
        kls, esk = load_data.cumulative_rescaling(make_copy_raw_dataset)
        load_data.rescale_inputs(make_copy_raw_dataset, kls, esk)
    chi2s = compute_exp_chi2(
        make_copy_raw_dataset,
        fit_dict["sf_model"],
        **runcard_content["fit_parameters"],
    )
    with open(f"{replica_dir}/fitinfo.json", "r+") as fstream:
        json_file = json.load(fstream)
        json_file.update({"exp_chi2s": chi2s})
        # Sets file's current position at offset.
        fstream.seek(0)
        json.dump(json_file, fstream, sort_keys=True, indent=4)
