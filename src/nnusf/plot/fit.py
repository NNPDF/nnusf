# -*- coding: utf-8 -*-
import json
import logging
import pathlib

import numpy as np
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..sffit.sum_rules import check_sumrule, effective_charge
from ..sffit.load_data import load_experimental_data
from ..sffit.load_fit_data import get_predictions_q, load_models
from .utils import plot_point_cov, format_jlab

_logger = logging.getLogger(__name__)
PARRENT_PATH = pathlib.Path(__file__).parents[1]
MPLSTYLE = PARRENT_PATH.joinpath("plotstyle.mplstyle")
plt.style.use(MPLSTYLE)

EQ_LABELS = {"GLS": r"$F_3$", "Bjorken": r"$F_1$"}

BASIS = [
    r"$F_2^\nu$",
    r"$F_L^\nu$",
    r"$xF_3^\nu$",
    r"$F_2^{\bar{\nu}}$",
    r"$F_L^{\bar{\nu}}$",
    r"$xF_3^{\bar{\nu}}$",
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


def _check_validity_models(models: list) -> None:
    """Check if a model actually exists.

    Parameters:
    -----------
    models: list
        list of models
    """
    if len(models) == 0:
        _logger.error("No model available")
        return


def save_figs(
    figure: Figure,
    filename: pathlib.Path,
    formats: list = [".png", ".pdf"],
    dpi=100,
) -> None:
    """Save all the figures into both PNG and PDF."""
    for format in formats:
        figure.savefig(
            str(filename) + format,
            bbox_inches="tight",
            dpi=dpi,
        )
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
    ax.scatter(
        tr_chi2s,
        vl_chi2s,
        s=60,
        marker="o",
        edgecolors="white",
        alpha=0.5,
    )
    ax.scatter(
        tr_chi2s.mean(),
        vl_chi2s.mean(),
        s=60,
        marker="s",
        color="C1",
        edgecolors="white",
    )
    plot_point_cov(
        np.array([tr_chi2s, vl_chi2s]).T,
        nstd=2,
        edgecolor="C1",
        facecolor="white",
        zorder=0,
        alpha=0.75,
        linewidth=0.75,
        label=r"$2\sigma~\rm{Ellipse}$",
    )
    ax.grid(color="grey", alpha=0.2, linewidth=0.5, zorder=0)
    ax.set_xlabel(r"$\chi^2_{\rm tr}$")
    ax.set_ylabel(r"$\chi^2_{\rm vl}$")
    ax.set_xlim([min_boundary, max_boundary])
    ax.set_ylim([min_boundary, max_boundary])
    ax.plot([0, 1], [0, 1], color="#3f7f93", transform=ax.transAxes)
    ax.legend()

    save_path = pathlib.Path(kwargs["output"]) / "chi2_split"
    save_figs(fig, save_path, dpi=350)


def smallx_exponent_distribution(**kwargs):
    """Plot the distribution of small-x exponents."""
    fitinfo = pathlib.Path(kwargs["fit"]).glob("replica_*/fitinfo.json")

    distr_epochs = []
    for repinfo in fitinfo:
        with open(repinfo, "r") as file_stream:
            jsonfile = json.load(file_stream)
        distr_epochs.append(np.asarray(jsonfile["small_x"]))
    distr_epochs = np.asarray(distr_epochs)

    fig, axs = plt.subplots(
        figsize=(3 * 5, 2 * 3),
        nrows=2,
        ncols=3,
        layout="constrained",
    )
    # Loop over the exponent of the structure function
    for index, sf_dist in enumerate(distr_epochs.T):
        sf_dist = sf_dist[sf_dist >= 0]
        binning = np.linspace(sf_dist.min(), sf_dist.max(), 10, endpoint=True)
        bar_width = binning[1] - binning[0]
        freq, bins = np.histogram(sf_dist, bins=binning, density=False)

        ax = axs.flat[index]
        center_bins = (bins[:-1] + bins[1:]) / 2
        ax.bar(
            center_bins,
            freq,
            width=bar_width,
            color="C0",
            edgecolor="C0",
            alpha=0.45,
            linewidth=1,
        )
        ax.axvline(x=sf_dist.mean(), lw=2, color="C1", label=r"$\rm{Mean}$")
        ax.axvline(
            x=sf_dist.mean() - sf_dist.std(), lw=0.75, ls="--", color="C1"
        )
        ax.axvline(
            x=sf_dist.mean() + sf_dist.std(), lw=0.75, ls="--", color="C1"
        )

        ax.text(0.88, 0.8, BASIS[index], size=16, transform=ax.transAxes)
        if index >= 3:
            ax.set_xlabel(r"$\alpha$")
        if index == 0 or index == 3:
            ax.set_ylabel(r"$\rm{Frequency}$")
        if index == 0:
            ax.legend()

    save_path = pathlib.Path(kwargs["output"]) / "smallx_exponent"
    save_figs(fig, save_path, dpi=350)


def training_epochs_distribution(**kwargs):
    fitinfo = pathlib.Path(kwargs["fit"]).glob("replica_*/fitinfo.json")

    tr_epochs = []
    for repinfo in fitinfo:
        with open(repinfo, "r") as file_stream:
            jsonfile = json.load(file_stream)
        tr_epochs.append(jsonfile["best_epochs"])
    tr_epochs = np.asarray(tr_epochs)
    tr_epochs_std = np.std(tr_epochs)
    binning = np.linspace(tr_epochs.min(), tr_epochs.max(), 10, endpoint=True)
    bar_width = binning[1] - binning[0]
    freq, bins = np.histogram(tr_epochs, bins=binning, density=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    center_bins = (bins[:-1] + bins[1:]) / 2
    ax.bar(
        center_bins,
        freq,
        width=bar_width,
        color="C0",
        edgecolor="C0",
        alpha=0.45,
        linewidth=1,
    )
    ax.axvline(x=tr_epochs.mean(), lw=2, color="C1", label=r"$\rm{Mean}$")
    ax.axvline(x=tr_epochs.mean() - tr_epochs_std, lw=0.75, ls="--", color="C1")
    ax.axvline(x=tr_epochs.mean() + tr_epochs_std, lw=0.75, ls="--", color="C1")
    ax.set_xlabel(r"$\rm{Epochs}$")
    ax.set_ylabel(r"$\rm{Frequency}$")
    ax.grid(color="grey", alpha=0.2, linewidth=0.5, zorder=0)
    ax.legend()

    save_path = pathlib.Path(kwargs["output"]) / "distr_epochs"
    save_figs(fig, save_path, dpi=350)


def check_sum_rules(**kwargs):
    q2grids, gls_results, preds_int = check_sumrule(**kwargs)

    preds_int_mean = np.mean(preds_int, axis=0)
    preds_int_stdev = np.std(preds_int, axis=0)

    fig, ax = plt.subplots()
    ax.errorbar(
        q2grids,
        preds_int_mean,
        yerr=preds_int_stdev,
        color="C1",
        fmt=".",
        marker="s",
        markersize=11,
        mfc="w",
        label=r"$\rm{NNSF}\nu$",
        capsize=6,
        zorder=0,
    )
    ax.scatter(
        q2grids,
        gls_results,
        color="C0",
        s=45,
        marker="o",
        label=r"$\rm{" + f"{kwargs['rule']}" + r"}$",
        zorder=1,
    )

    xmin_log = abs(kwargs["nx_specs"]["xmin_log"])
    xmin_label = r"$10^{" + f"-{xmin_log}" + r"}$"

    ax.grid(alpha=0.1)
    ax.legend(
        title=rf"$A={kwargs['a_value']}$" + r",~$x_{\rm min}=$" + xmin_label
    )
    ax.set_xlabel(r"$Q^2~[\rm{GeV}^2]$")
    ax.set_ylabel(r"$\rm{Value}$")

    plotname = f"{kwargs['rule'].lower()}_sumrule_a{kwargs['a_value']}_xmin{xmin_log}"
    save_path = pathlib.Path(kwargs["output"]) / plotname

    save_figs(fig, save_path)


def check_effective_charge(**kwargs):
    # Compute a_eff as a function of Q2 using NN model
    q2grids, preds_int = effective_charge(**kwargs)

    # Load the experimental values including uncertainties
    q_values, central, error = format_jlab(data="JLAB")

    fig, ax = plt.subplots()
    ax.errorbar(
        q_values,
        central / np.pi,
        yerr=error,
        color="C0",
        # fmt=".",
        marker="o",
        markersize=11,
        mfc="w",
        label=r"$\rm{JLAB~(EG1,EG4,E97110)}$",
        capsize=6,
        zorder=0,
        linestyle="none",
    )

    # Compute the 68% Confidence Level for NN predictions
    lower_68 = np.sort(preds_int, axis=0)[int(.16 * preds_int.shape[0])]
    upper_68 = np.sort(preds_int, axis=0)[int(.84 * preds_int.shape[0])]
    mean_prd = np.mean(preds_int, axis=0)
    ax.fill_between(
        np.sqrt(q2grids),
        lower_68,
        upper_68,
        color="C1",
        alpha=0.15,
        label=r"$\rm{NNSF}\nu$",
    )
    ax.plot(np.sqrt(q2grids), mean_prd, color="C1", lw=1.5)

    xmin_log = abs(kwargs["nx_specs"]["xmin_log"])
    xmin_label = r"$10^{" + f"-{xmin_log}" + r"}$"

    ax.grid(alpha=0.1)
    ax.legend(
        title=rf"$A={kwargs['a_value']}$" + r",~$x_{\rm min}=$" + xmin_label
    )
    ax.set_xlim(left=np.sqrt(q2grids).min(), right=np.sqrt(q2grids).max())
    ax.set_xlabel(r"$Q~[\rm{GeV}]$")
    ax.set_ylabel(r"$\alpha_{\rm eff}$" + f"({EQ_LABELS[kwargs['rule']]})")

    plotname = f"{kwargs['rule'].lower()}_aseff_a{kwargs['a_value']}_xmin{xmin_log}"
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
        ax.set_ylabel(BASIS[prediction_index])
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
        ax.set_ylabel(BASIS[prediction_index])
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
    """Producing plots comparing the data (experimental & Yadism) with
    the Neural Network Predictions.
    """
    models = load_models(**kwargs)
    _logger.info("Models successfully loaded.")
    _check_validity_models(models)

    # Load the datasets all at once in order to rescale
    raw_datasets, datasets = load_experimental_data(
        kwargs["experiments"],
        input_scaling=kwargs.get("rescale_inputs", None),
        kincuts=kwargs.get("kinematic_cuts", {}),
        verbose=False,
    )
    # Copy the dataset kinematics regardless of scaling
    copy_kins = {k: v.kinematics for k, v in raw_datasets.items()}

    count_plots = 0
    for experiment, data in datasets.items():
        _logger.info(f"'[{experiment:<25}]' Plotting NN predictions vs Data.")
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
