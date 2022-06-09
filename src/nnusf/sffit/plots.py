from load_fit_data import get_predictions_q
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pathlib

basis = [
    r"$F_2$",
    r"$F_L$",
    r"$xF_3$",
    r"$\bar{F}_2$",
    r"$\bar{F}_L$",
    r"$x\bar{F}_3$",
]


def plot_sfs_q(*args, **kwargs):
    prediction_info = get_predictions_q(**kwargs)
    predictions = prediction_info.predictions
    q_grid = prediction_info.q
    n_sfs = prediction_info.n_sfs
    lower_68 = np.sort(predictions, axis=0)[int(0.16 * n_sfs)]
    upper_68 = np.sort(predictions, axis=0)[int(0.84 * n_sfs)]
    mean_sfs = np.mean(predictions, axis=0)
    std_sfs = np.std(predictions, axis=0)
    for prediction_index in range(predictions.shape[2]):
        fig, ax = plt.subplots()
        ax.set_xlabel("Q (GeV)")
        ax.set_ylabel(basis[prediction_index])
        ax.set_title(f"x={prediction_info.x}, A={prediction_info.A}")
        prediction = predictions[:, :, prediction_index]
        ax.plot(
            q_grid, lower_68[:, prediction_index], color="C0", linestyle="--"
        )
        ax.plot(
            q_grid, upper_68[:, prediction_index], color="C0", linestyle="--"
        )
        ax.fill_between(
            q_grid,
            mean_sfs[:, prediction_index] + std_sfs[:, prediction_index],
            mean_sfs[:, prediction_index] - std_sfs[:, prediction_index],
            color="C0",
            alpha=0.4,
        )
        for replica_prediction in prediction:
            ax.plot(q_grid, replica_prediction, color="C0")
        savepath = (
            pathlib.Path(kwargs["output"]) / f"plot_{prediction_index}.png"
        )
        fig.savefig(savepath)


def prediction_data_comparison(*args, **kwargs):
    pass
