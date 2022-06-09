import pathlib
import tensorflow as tf
import numpy as np
from dataclasses import dataclass

@dataclass
class PredictionInfo:
    predictions: np.ndarray
    q: np.ndarray
    x: np.ndarray
    A: np.ndarray


def load_models(fit):
    path_to_fit_folder = pathlib.Path(fit)
    models = []
    for replica_folder in path_to_fit_folder.rglob("replica_*/"):
        model_folder = replica_folder / "model"
        models.append(tf.keras.models.load_model(model_folder, compile=False))
    return models


def get_predictions_q(
    fit, a_slice=26, x_slice=0.01, qmin=1e-1, qmax=5, *args, **kwargs
):
    "ouputs a PredicitonInfo object for fixed A and x"
    del args
    del kwargs

    q_values = np.linspace(start=qmin, stop=qmax)
    # the additional [] are go get the correct input shape
    inputs = tf.constant([[[x_slice, q, a_slice] for q in q_values]])

    models = load_models(fit)

    predictions = []
    for model in models:
        # [0] to cut the superflous dimension
        predictions.append(model.predict(inputs)[0])

    prediction_info = PredictionInfo(
        predictions=np.array(predictions),
        x=x_slice,
        A=a_slice,
        q=q_values,
    )

    return prediction_info
