# -*- coding: utf-8 -*-
"""Executable to perform the structure function fit."""
import json
import logging
import pathlib
import shutil
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
    # Create a folder for the replica
    if destination is None:
        destination = pathlib.Path.cwd().absolute() / runcard.stem

    if destination.exists():
        _logger.warning(f"{destination} already exists, overwriting content.")

    replica_dir = destination / f"replica_{replica}"
    replica_dir.mkdir(parents=True, exist_ok=True)

    # copy runcard to the fit folder
    shutil.copy(runcard, replica_dir.parent / "runcard.yml")

    # load runcard
    runcard_content = yaml.safe_load(runcard.read_text())
    experiments_dict = runcard_content["experiments"]

    # Set global seeds
    set_global_seeds(global_seed=runcard_content["global_seeds"] + replica)

    # Instantiate class that loads the datasets
    data_info = load_data.load_experimental_data(experiments_dict)
    # create pseudodata and add it to the data_info object
    load_data.add_pseudodata(data_info)
    # create a training mask and add it to the data_info object
    load_data.add_tr_filter_mask(data_info, runcard_content["trvlseed"])

    fit_dict = generate_models(data_info, **runcard_content["fit_parameters"])

    # Compile the training and validationa nd perform the fit
    resdic = perform_fit(
        fit_dict, data_info, replica_dir, **runcard_content["fit_parameters"]
    )

    # Store the models in the relevant replica subfolders
    final_placeholder = tf.keras.layers.Input(shape=(None, 3))
    saved_model = tf.keras.Model(
        inputs=final_placeholder,
        outputs=fit_dict["sf_model"](final_placeholder),
    )
    saved_model.save(replica_dir / "model")

    # Store the metadata in the relevant replicas subfodlers
    with open(f"{replica_dir}/fitinfo.json", "w", encoding="UTF-8") as ostream:
        json.dump(resdic, ostream, sort_keys=True, indent=4)
