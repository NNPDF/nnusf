# -*- coding: utf-8 -*-
import logging
import pathlib
from dataclasses import dataclass
from typing import Union

import numpy as np
import tensorflow as tf
import yaml

_logger = logging.getLogger(__name__)


@dataclass
class PredictionInfo:
    q: np.ndarray
    x: Union[int, float, list]
    A: int
    n_sfs: int
    predictions: Union[np.ndarray, list]


def input_scaling(input_arr, map_from, map_to):
    scaled_inputs = []
    for index, kin_var in enumerate(input_arr.T):
        input_scaling = np.interp(kin_var, map_from[index], map_to[index])
        scaled_inputs.append(input_scaling)
    return np.array(scaled_inputs).T


def load_models(fit, **kwargs):
    del kwargs
    path_to_fit_folder = pathlib.Path(fit)
    models = []
    for replica_folder in path_to_fit_folder.rglob("replica_*/"):
        model_folder = replica_folder / "model"
        models.append(tf.keras.models.load_model(model_folder, compile=False))
    return models


def get_predictions_q(
    fit, a_slice=26, x_slice=[0.01], qmin=1e-1, qmax=5, *args, **kwargs
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

    q_values = np.linspace(start=qmin, stop=qmax)
    # the additional [] are go get the correct input shape
    if isinstance(x_slice, (int, float)):
        input_list = [[x_slice, q, a_slice] for q in q_values]
    elif isinstance(x_slice, list):
        input_list = [[x, q, a_slice] for x in x_slice for q in q_values]
    else:
        raise ValueError("The value of x is of an unrecognised type.")

    if fitcard.get("rescale_inputs", None):
        input_list = input_scaling(
            np.array(input_list),
            fitcard["scaling"]["map_from"],
            fitcard["scaling"]["map_to"],
        )
        _logger.info("Inputs are first being rescaled.")

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

    prediction_info = PredictionInfo(
        predictions=predictions,
        x=x_slice,
        A=a_slice,
        q=q_values,
        n_sfs=len(predictions),
    )

    return prediction_info
