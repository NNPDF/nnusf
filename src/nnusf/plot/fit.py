# -*- coding: utf-8 -*-
import json
import logging
import pathlib

import numpy as np
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..data.loader import Loader
from ..sffit.load_data import path_to_coefficients, path_to_commondata
from ..sffit.load_fit_data import get_predictions_q, load_models

_logger = logging.getLogger(__name__)
PARRENT_PATH = pathlib.Path(__file__).parents[1]
MPLSTYLE = PARRENT_PATH.joinpath("plotstyle.mplstyle")
plt.style.use(MPLSTYLE)

basis = [
    r"$F_2$",
    r"$F_L$",
    r"$xF_3$",
    r"$\bar{F}_2$",
    r"$\bar{F}_L$",
    r"$x\bar{F}_3$",
]

MAP_OBS_LABEL = {
    "F2": r"$F_2$",
    "FW": r"$F_W$",
    "F3": r"$xF_3$",
    "DXDYNUU": r"$d^2\sigma^{\nu}/(dxdQ^2)$",
    "DXDYNUB": r"$d^2\sigma^{\bar{\nu}}/(dxdQ^2)$",
}


class InputError(Exception):
    pass


def save_figs(
    figure: Figure, filename: pathlib.Path, formats: list = [".png", ".pdf"]
) -> None:
    """Save all the figures into both PNG and PDF."""
    for format in formats:
        figure.savefig(str(filename) + format)
    plt.close(figure)


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


def training_validation_split(**kwargs):
    fitinfo = pathlib.Path(kwargs["fit"]).glob("replica_*/fitinfo.json")
    tr_chi2s, vl_chi2s = [], []

    for repinfo in fitinfo:
        with open(repinfo, "r") as file_stream:
            jsonfile = json.load(file_stream)
        tr_chi2s.append(jsonfile["best_tr_chi2"])
        vl_chi2s.append(jsonfile["best_vl_chi2"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tr_chi2s, vl_chi2s, s=20, marker="s")
    ax.set_xlabel(r"$\chi^2_{\rm tr}$")
    ax.set_ylabel(r"$\chi^2_{\rm vl}$")
    save_path = pathlib.Path(kwargs["output"]) / "chi2_split"
    save_figs(fig, save_path)


def sfs_q_replicas(**kwargs):
    prediction_info = get_predictions_q(**kwargs)
    predictions = prediction_info.predictions
    if not isinstance(predictions, np.ndarray):
        raise InputError("The input x should be a float.")
    q_grid = prediction_info.q
    for prediction_index in range(predictions.shape[2]):
        fig, ax = plt.subplots()
        ax.set_xlabel("Q2 (GeV)")
        ax.set_ylabel(basis[prediction_index])
        ax.set_title(f"x={prediction_info.x}, A={prediction_info.A}")
        prediction = predictions[:, :, prediction_index]
        for replica_prediction in prediction:
            ax.plot(q_grid, replica_prediction, color="C0")
        savepath = (
            pathlib.Path(kwargs["output"])
            / f"plot_sfs_q_replicas_{prediction_index}"
        )
        save_figs(fig, savepath)


def sf_q_band(**kwargs):
    prediction_info = get_predictions_q(**kwargs)
    predictions = prediction_info.predictions
    if not isinstance(predictions, np.ndarray):
        raise InputError("The input x should be a float.")
    q_grid = prediction_info.q
    n_sfs = prediction_info.n_sfs
    lower_68 = np.sort(predictions, axis=0)[int(0.16 * n_sfs)]
    upper_68 = np.sort(predictions, axis=0)[int(0.84 * n_sfs)]
    mean_sfs = np.mean(predictions, axis=0)
    std_sfs = np.std(predictions, axis=0)
    for prediction_index in range(predictions.shape[2]):
        fig, ax = plt.subplots()
        ax.set_xlabel("Q2 (GeV)")
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
            pathlib.Path(kwargs["output"])
            / f"plot_sf_q_band_{prediction_index}"
        )
        save_figs(fig, savepath)


def save_predictions_txt(**kwargs):
    predinfo = get_predictions_q(**kwargs)
    pred = predinfo.predictions
    q2_grids = predinfo.q
    xval = predinfo.x
    # Make sure that everything is a list
    pred = [pred] if not isinstance(pred, list) else pred
    xval = [xval] if not isinstance(xval, list) else xval
    q2_grids = q2_grids[np.newaxis, :]

    # Loop over the different values of x
    stacked_results = []
    for idx, pr in enumerate(pred):
        predshape = pr[:, :, 0].shape
        broad_xvalues = np.broadcast_to(xval[idx], predshape)
        broad_qvalues = np.broadcast_to(q2_grids, predshape)
        # Construct the replica index array
        repindex = np.arange(pr.shape[0])[:, np.newaxis]
        repindex = np.broadcast_to(repindex, predshape)
        # Stack all the arrays together
        stacked_list = [repindex, broad_xvalues, broad_qvalues]
        stacked_list += [pr[:, :, i] for i in range(pr.shape[-1])]
        stacked = np.stack(stacked_list).reshape((9, -1)).T
        stacked_results.append(stacked)
    predictions = np.concatenate(stacked_results, axis=0)
    np.savetxt(
        f"{pathlib.Path(kwargs['output'])}/sfs_{predinfo.A}.txt",
        predictions,
        header=f"replica x Q2 F2nu FLnu xF3nu F2nub FLnub xF3nub",
        fmt="%d %e %e %e %e %e %e %e %e",
    )


def prediction_data_comparison(**kwargs):
    models = load_models(**kwargs)
    _logger.info("Models successfully loaded.")
    if len(models) == 0:
        _logger.error("No model available")
        return

    count_plots = 0
    for experiment in kwargs["experiments"]:
        obs_label = MAP_OBS_LABEL[experiment.split("_")[-1]]
        expt_name = experiment.split("_")[0]
        data = Loader(experiment, path_to_commondata, path_to_coefficients)
        kinematics = data.kinematics
        coefficients = data.coefficients
        observable_predictions = []
        for model in models:
            kins = np.expand_dims(
                data.kinematics, axis=0
            )  # add batch dimension
            prediction = model(kins)
            prediction = prediction[0]  # remove batch dimension
            observable_predictions.append(
                tf.einsum("ij,ij->i", prediction, coefficients)
            )
        observable_predictions = np.array(observable_predictions)
        mean_observable_predictions = observable_predictions.mean(axis=0)
        std_observable_predictions = observable_predictions.std(axis=0)
        for x_slice in np.unique(kinematics[:, 0]):
            fig, ax = plt.subplots()
            ax.set_title(rf"{expt_name}:~$A$={kinematics[0,2]}, $x$={x_slice}")
            mask = np.where(kinematics[:, 0] == x_slice)[0]
            tmp_kinematics = kinematics[mask]
            diag_covmat = np.diag(data.covmat)[mask]
            ax.errorbar(
                tmp_kinematics[:, 1],
                data.central_values[mask],
                yerr=np.sqrt(diag_covmat),
                fmt=".",
                label="Data",
                capsize=5,
            )
            ax.errorbar(
                tmp_kinematics[:, 1],
                mean_observable_predictions[mask],
                yerr=std_observable_predictions[mask],
                fmt=".",
                label="NN Predictions",
                capsize=5,
            )
            ax.set_xlabel(r"$Q^2~[\mathrm{GeV}^2]$")
            ax.set_ylabel(f"{obs_label}" + r"$~(x, Q^2)$")
            ax.legend()
            savepath = (
                pathlib.Path(kwargs["output"])
                / f"prediction_data_comparison_{count_plots}"
            )
            count_plots += 1
            save_figs(fig, savepath)


def chi2_history_plot(xmin=None, **kwargs):
    fitpath = kwargs["fit"]
    outputpath = kwargs["output"]

    fit_folder = pathlib.Path(fitpath)
    count_plots = 0
    for foldercontent in fit_folder.iterdir():
        if "replica_" in foldercontent.name:
            chi2_history_file = foldercontent / "chi2_history.json"
            if chi2_history_file.exists():
                with open(chi2_history_file, "r") as f:
                    data = json.load(f)
                epochs = [int(i) for i in data.keys()]
                vl_chi2 = [i["vl"] for i in data.values()]
                tr_chi2 = [i["tr"] for i in data.values()]
                count_plots += 1
                fig, ax = plt.subplots()
                ax.set_title(f"replica {foldercontent.name.split('_')[1]}")
                ax.set_xlabel("epoch")
                ax.set_ylabel("loss")
                if xmin != None:
                    index_cut = epochs.index(xmin)
                    epochs = epochs[index_cut:]
                    vl_chi2 = vl_chi2[index_cut:]
                    tr_chi2 = tr_chi2[index_cut:]
                ax.plot(epochs, vl_chi2, label="validation")
                ax.plot(epochs, tr_chi2, label="training")
                ax.legend()
                savepath = (
                    pathlib.Path(outputpath)
                    / f"chi2_history_plot_{count_plots}"
                )
                save_figs(fig, savepath)
