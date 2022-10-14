# -*- coding: utf-8 -*-
import logging
import pathlib
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Union

import numpy as np
import tensorflow as tf
import yaml

from .load_data import construct_expdata_instance
from .scaling import cumulative_rescaling, kinematics_mapping

_logger = logging.getLogger(__name__)


@dataclass
class PredictionInfo:
    q: np.ndarray
    x: Union[int, float, list]
    A: int
    n_sfs: int
    predictions: Union[np.ndarray, list]


def load_single_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


def load_models_parallel(fit, **kwargs):
    del kwargs
    path_to_fit_folder = pathlib.Path(fit)
    models = [m / "model" for m in path_to_fit_folder.rglob("replica_*/")]
    pool = Pool(processes=10)
    loaded_models = pool.map(load_single_model, models)
    return loaded_models


def load_models(fit, **kwargs):
    del kwargs
    path_to_fit_folder = pathlib.Path(fit)
    models = []
    for replica_folder in path_to_fit_folder.rglob("replica_*/"):
        model_folder = replica_folder / "model"
        models.append(tf.keras.models.load_model(model_folder, compile=False))
    return models


def get_predictions_q(
    fit,
    a_slice=26,
    x_slice=[0.01],
    q2min=1e-1,
    q2max=5,
    nq2p=100,
    *args,
    **kwargs
):
    """ouputs a PredicitonInfo object for fixed A and x.

    Parameters:
    -----------
    fit: pathlib.Path
        Path to the fit folder
    """
    del args
    del kwargs

    fitting_card = pathlib.Path(fit).joinpath("runcard.yml")
    fitcard = yaml.load(fitting_card.read_text(), Loader=yaml.Loader)

    q_values = np.linspace(start=q2min, stop=q2max, num=nq2p)
    # the additional [] are go get the correct input shape
    if isinstance(x_slice, (int, float)):
        input_list = [[x_slice, q, a_slice] for q in q_values]
    elif isinstance(x_slice, list):
        input_list = [[x, q, a_slice] for x in x_slice for q in q_values]
    else:
        raise ValueError("The value of x is of an unrecognised type.")

    if fitcard.get("rescale_inputs", None):
        unscaled_datainfo = construct_expdata_instance(
            experiment_list=fitcard["experiments"],
            kincuts=fitcard.get("kinematics_cuts", {}),
        )
        map_from, map_to = cumulative_rescaling(unscaled_datainfo)
        transp_inputs = np.array(input_list).T
        scaled = kinematics_mapping(transp_inputs, map_from, map_to)
        input_list = np.array(scaled).T
        _logger.warning("Input kinematics are being scaled.")

    input_kinematics = [input_list]
    inputs = tf.constant(input_kinematics)

    # Load the models and perform predictions
    models = load_models(fit)
    predictions = []
    for model in models:
        # [0] to cut the superflous dimension
        predictions.append(model.predict(inputs)[0])

    # Check if we need to split the predictions
    predictions = np.array(predictions)
    if isinstance(x_slice, list):
        nbsplit = predictions.shape[1] // q_values.shape[0]
        predictions = np.split(predictions, nbsplit, axis=1)

    # TODO: fix value of n_sfs
    prediction_info = PredictionInfo(
        predictions=predictions,
        x=x_slice,
        A=a_slice,
        q=q_values,
        n_sfs=len(predictions),
    )

    return prediction_info
