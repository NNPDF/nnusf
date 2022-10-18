# -*- coding: utf-8 -*-
import json
import logging
import pathlib

import numpy as np
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..sffit.check_gls import check_gls_sumrules
from ..sffit.load_data import load_experimental_data
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
    "F2_MATCHING": r"$F_2^{\rm M}$",
    "FW": r"$F_W$",
    "FW_MATCHING": r"$F_W^{\rm M}$",
    "F3": r"$xF_3$",
    "F3_MATCHING": r"$xF_3^{\rm M}$",
    "DXDYNUU": r"$d^2\sigma^{\nu}/(dxdQ^2)$",
    "DXDYNUU_MATCHING": r"$d^2\sigma^{\nu, \rm{M}}/(dxdQ^2)$",
    "DXDYNUB": r"$d^2\sigma^{\bar{\nu}}/(dxdQ^2)$",
    "DXDYNUB_MATCHING": r"$d^2\sigma^{\bar{\nu}, \rm{M}}/(dxdQ^2)$",
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


def training_validation_split(**kwargs):
    fitinfo = pathlib.Path(kwargs["fit"]).glob("replica_*/fitinfo.json")
    tr_chi2s, vl_chi2s = [], []

    for repinfo in fitinfo:
        with open(repinfo, "r") as file_stream:
            jsonfile = json.load(file_stream)
        tr_chi2s.append(jsonfile["best_tr_chi2"])
        vl_chi2s.append(jsonfile["best_vl_chi2"])
    tr_chi2s, vl_chi2s = np.asarray(tr_chi2s), np.asarray(vl_chi2s)
    min_boundary = np.min([tr_chi2s, vl_chi2s]) - 0.05
    max_boundary = np.max([tr_chi2s, vl_chi2s]) + 0.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tr_chi2s, vl_chi2s, s=30, marker="s")
    ax.scatter(tr_chi2s.mean(), vl_chi2s.mean(), s=30, marker="s", color="C1")
    ax.set_xlabel(r"$\chi^2_{\rm tr}$")
    ax.set_ylabel(r"$\chi^2_{\rm vl}$")
    ax.set_xlim([min_boundary, max_boundary])
    ax.set_ylim([min_boundary, max_boundary])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)

    save_path = pathlib.Path(kwargs["output"]) / "chi2_split"
    save_figs(fig, save_path)


def training_epochs_distribution(**kwargs):
    fitinfo = pathlib.Path(kwargs["fit"]).glob("replica_*/fitinfo.json")

    tr_epochs = []
    for repinfo in fitinfo:
        with open(repinfo, "r") as file_stream:
            jsonfile = json.load(file_stream)
        tr_epochs.append(jsonfile["best_epochs"])
    tr_epochs = np.asarray(tr_epochs)
    binning = np.linspace(tr_epochs.min(), tr_epochs.max(), 10, endpoint=True)
    bar_width = binning[1] - binning[0]
    freq, bins = np.histogram(tr_epochs, bins=binning, density=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    center_bins = (bins[:-1] + bins[1:]) / 2
    ax.bar(center_bins, freq, width=bar_width)
    ax.axvline(x=tr_epochs.mean(), lw=2, color="C1")
    ax.set_xlabel(r"$\rm{Epochs}$")
    ax.set_ylabel(r"$\rm{Frequency}$")

    save_path = pathlib.Path(kwargs["output"]) / "distr_epochs"
    save_figs(fig, save_path)


def gls_sum_rules(**kwargs):
    q2grids, gls_results, xf3avg_int = check_gls_sumrules(**kwargs)

    xf3avg_int_mean = np.mean(xf3avg_int, axis=0)
    xf3avg_int_stdev = np.std(xf3avg_int, axis=0)

    fig, ax = plt.subplots()
    ax.scatter(q2grids, gls_results, color="C0", s=20, marker="s", label="GLS")
    ax.errorbar(
        q2grids,
        xf3avg_int_mean,
        yerr=xf3avg_int_stdev,
        color="C1",
        fmt=".",
        label="NN Predictions",
        capsize=5,
    )
    ax.legend(title=f"Comparison for A={kwargs['a_value']}")
    ax.set_xlabel(r"$Q^2~[\rm{GeV}^2]$")
    ax.set_ylabel(r"$\rm{Value}$")
    plotname = f"gls_sumrule_a{kwargs['a_value']}"
    save_path = pathlib.Path(kwargs["output"]) / plotname
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
        ax.set_xlabel(r"$Q^2~[\mathrm{GeV^2}]$")
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
        ax.set_xscale("log")
        savepath = (
            pathlib.Path(kwargs["output"])
            / f"sf_q_band_{prediction_index}_A{prediction_info.A}"
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

    # Load the datasets all at once in order to rescale
    raw_datasets, datasets = load_experimental_data(
        kwargs["experiments"],
        input_scaling=kwargs.get("rescale_inputs", None),
        kincuts=kwargs.get("kinematic_cuts", {}),
    )
    # Copy the dataset kinematics regardless of scaling
    copy_kins = {k: v.kinematics for k, v in raw_datasets.items()}

    count_plots = 0
    for experiment, data in datasets.items():
        _logger.info(f"Plotting data vs. NN for '{experiment}'")
        if "_MATCHING" not in experiment:
            obsname = experiment.split("_")[-1]
        else:
            obsname = experiment.split("_")[-2] + "_MATCHING"

        obs_label = MAP_OBS_LABEL[obsname]
        expt_name = experiment.split("_")[0]

        kinematics = copy_kins[experiment]
        observable_predictions = []
        for model in models:
            kins = np.expand_dims(
                data.kinematics, axis=0
            )  # add batch dimension
            prediction = model(kins)
            prediction = prediction[0]  # remove batch dimension
            observable_predictions.append(
                tf.einsum("ij,ij->i", prediction, data.coefficients)
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
            chi2_history_file = foldercontent / "chi2_history.yaml"
            if chi2_history_file.exists():
                data = yaml.safe_load(chi2_history_file.read_text())
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
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend()
                savepath = (
                    pathlib.Path(outputpath)
                    / f"chi2_history_plot_{count_plots}"
                )
                save_figs(fig, savepath)


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
