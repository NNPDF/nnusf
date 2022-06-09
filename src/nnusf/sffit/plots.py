from load_fit_data import get_predictions_q
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pathlib

basis = [
    r"$F_2$",
    r"$F_L$",
    r"$F_3$",
    r"$\bar{F}_2$",
    r"$\bar{F}_L$",
    r"$\bar{F}_3$",
]


def plot_sfs_q(*args, **kwargs):
    prediction_info = get_predictions_q(**kwargs)
    predictions = prediction_info.predictions
    q_grid = prediction_info.q
    for prediction_index in range(predictions.shape[2]):
        fig, ax = plt.subplots()
        ax.set_xlabel("Q (GeV)")
        ax.set_ylabel(basis[prediction_index])
        ax.set_title(f"x={prediction_info.x}, A={prediction_info.A}")
        prediction = predictions[:, :, prediction_index]
        for replica_prediction in prediction:
            ax.plot(q_grid, replica_prediction, color="C0")
        savepath = (
            pathlib.Path(kwargs["output"]) / f"plot_{prediction_index}.png"
        )
        fig.savefig(savepath)
