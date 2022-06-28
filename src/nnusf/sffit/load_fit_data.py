# -*- coding: utf-8 -*-
import pathlib
import numpy as np
import tensorflow as tf

from dataclasses import dataclass
from typing import Union


@dataclass
class PredictionInfo:
    q: np.ndarray
    x: Union[int, float, list]
    A: int
    n_sfs: int
    predictions: Union[np.ndarray, list]


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
    "ouputs a PredicitonInfo object for fixed A and x"
    del args
    del kwargs

    q_values = np.linspace(start=qmin, stop=qmax)
    # the additional [] are go get the correct input shape
    if isinstance(x_slice, (int, float)):
        input_list = [[[x_slice, q, a_slice] for q in q_values]]
    elif isinstance(x_slice, list):
        input_list = [[[x, q, a_slice] for x in x_slice for q in q_values]]
    else:
        raise ValueError("The value of x is of an unrecognised type.")
    inputs = tf.constant(input_list)

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
