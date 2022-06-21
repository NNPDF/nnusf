import logging
import pathlib

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import yaml

from ..data.loader import Loader
from ..sffit.load_data import path_to_coefficients, path_to_commondata
from ..sffit.load_fit_data import load_models, get_predictions_q

_logger = logging.getLogger(__name__)

basis = [
    r"$F_2$",
    r"$F_L$",
    r"$xF_3$",
    r"$\bar{F}_2$",
    r"$\bar{F}_L$",
    r"$x\bar{F}_3$",
]


def main(model: pathlib.Path, runcard: pathlib.Path, output: pathlib.Path):
    if output.exists():
        _logger.warning(f"{output} already exists, overwriting content.")
    output.mkdir(parents=True, exist_ok=True)

    runcard_content = yaml.safe_load(runcard.read_text())
    runcard_content["fit"] = str(model.absolute())
    runcard_content["output"] = str(output.absolute())

    for action in runcard_content["actions"]:
        func = globals()[action]
        func(**runcard_content)


def sfs_q_replicas(**kwargs):
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
            pathlib.Path(kwargs["output"])
            / f"plot_sfs_q_replicas_{prediction_index}.png"
        )
        fig.savefig(savepath)


def sf_q_band(**kwargs):
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
        ax.plot(
            q_grid,
            mean_sfs[:, prediction_index] - std_sfs[:, prediction_index],
            color="C0",
            linestyle="--",
        )
        ax.plot(
            q_grid,
            mean_sfs[:, prediction_index] + std_sfs[:, prediction_index],
            color="C0",
            linestyle="--",
        )
        ax.fill_between(
            q_grid,
            lower_68[:, prediction_index],
            upper_68[:, prediction_index],
            color="C0",
            alpha=0.4,
        )
        savepath = (
            pathlib.Path(kwargs["output"]) / f"plot_sf_q_band_{prediction_index}.png"
        )
        fig.savefig(savepath)


def prediction_data_comparison(**kwargs):
    models = load_models(**kwargs)
    if len(models) == 0:
        _logger.error("No model available")
        return

    count_plots = 0
    for experiment in kwargs["experiments"]:
        data = Loader(experiment, path_to_commondata, path_to_coefficients)
        kinematics = data.kinematics
        coefficients = data.coefficients
        observable_predictions = []
        for model in models:
            prediction = model(data.kinematics)
            observable_predictions.append(
                tf.einsum("ij,ij->i", prediction, coefficients)
            )
        observable_predictions = np.array(observable_predictions)
        mean_observable_predictions = observable_predictions.mean(axis=0)
        std_observable_predictions = observable_predictions.std(axis=0)
        for x_slice in np.unique(kinematics[:, 0]):
            fig, ax = plt.subplots()
            ax.set_title(f"{experiment}: A={kinematics[0,2]}, x={x_slice}")
            mask = np.where(kinematics[:, 0] == x_slice)[0]
            tmp_kinematics = kinematics[mask]
            diag_covmat = np.diag(data.covmat)[mask]
            ax.errorbar(
                tmp_kinematics[:, 1],
                data.central_values[mask],
                yerr=np.sqrt(diag_covmat),
                fmt=".",
                color="black",
                capsize=5,
            )
            ax.errorbar(
                tmp_kinematics[:, 1],
                mean_observable_predictions[mask],
                yerr=std_observable_predictions[mask],
                fmt=".",
                color="C0",
                capsize=5,
            )
            savepath = (
                pathlib.Path(kwargs["output"])
                / f"prediction_data_comparison_{count_plots}.png"
            )
            count_plots += 1
            fig.savefig(savepath, dpi=300)
            plt.close(fig)
