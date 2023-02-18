# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import copy
import logging
import pathlib
from typing import Optional

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from matplotlib import colors

from .. import utils
from ..data import loader
from . import utils as putils

matplotlib.use("Agg")

_logger = logging.getLogger(__file__)


_EXP_ORD = [
    "BEBCWA59",
    "NUTEV",
    "CHARM",
    "CDHSW",
    "CCFR",
    "CHORUS",
]


def compute(
    name: str,
    datapath: pathlib.Path,
    inverse: bool = False,
    norm: bool = True,
    cuts: Optional[dict[str, dict[str, float]]] = None,
) -> np.ndarray:
    """Compute covmat.

    Parameters
    ----------
    name: str
        name of the requested dataset
    datapath: pathlib.Path
        path to commondata
    inverse: bool
        if `True`, compute and plot the inverse of the covariance matrix
        (default: `False`)
    norm: bool
        if `True`, normalize the covariance matrix with central values (default:
        `True`)
    cuts: dict
        kinematic cuts

    Returns
    -------
    np.ndarray
        (inverse) covariance matrix computed

    """
    data = loader.Loader(name, datapath)
    covmat = data.covariance_matrix
    cv = data.central_values

    if cuts is not None:
        mask = putils.cuts(cuts, data.table)
        cv = cv[mask]
        covmat = covmat[mask][:, mask]

        _logger.info(f"Following cuts applied: {cuts}")

    if norm:
        covmat = covmat / cv / cv[:, np.newaxis]

    if inverse:
        covmat = np.linalg.inv(covmat)

    return covmat


def order_dict_experiment(dataspecs: dict, ordering: list = _EXP_ORD) -> dict:
    """Ordering a dictionary with respect to name of the experiments.

    Parameters
    ----------
    dataspecs: dict
        dictinary containing the specs of the covmat per datasets
    ordering: list
        list of names from which the dictionary will be ordered


    Returns
    -------
    dict:
        ordered dictionary
    """
    ordered_dict = {}
    for name in ordering:
        for key, value in dataspecs.items():
            if name in key:
                ordered_dict[key] = value
    return ordered_dict


def correlation_matrix(covmat: np.ndarray) -> np.ndarray:
    sqrt_diags = np.sqrt(np.diag(covmat))
    return covmat / sqrt_diags[:, np.newaxis] / sqrt_diags


def construct_ticklabels(dataspecs: dict) -> dict:
    # Initialize number of data points to be zero as placeholders
    new_dic = {k.split("_")[0]: 0 for k in dataspecs.keys()}

    # Construct a dict that stores the number of data points per exp
    for key, value in dataspecs.items():
        expname = key.split("_")[0]
        new_dic[expname] += value.shape[0]

    # Compute the relative center of the ticks
    relative_ticklabels = []
    number_datapoints = [0] + [v for v in new_dic.values()]
    ticklabels_temp = copy.deepcopy(number_datapoints)

    for i in range(1, len(number_datapoints)):
        number_datapoints[i] += number_datapoints[i - 1]
        rticks = number_datapoints[i - 1] + (ticklabels_temp[i] // 2)
        relative_ticklabels.append(rticks)

    tick_specs = {
        "tick_labels": list(new_dic.keys()),
        "tick_locs": relative_ticklabels,
        "separation_lines": number_datapoints,
    }

    return tick_specs


def heatmap(
    covmat: np.ndarray, title: str, specs: Optional[dict] = None
) -> matplotlib.figure.Figure:
    """Plot covariance matrix.

    Parameters
    ----------
    covmat: np.ndarray
        covariance matrix to plot
    symlog: bool
        if `True`, plot in symmetric logarithmic color scale

    Returns
    -------
    matplotlib.figure.Figure
        plotted figure

    """
    fig, ax = plt.subplots(figsize=(12, 12))
    matrixplot = ax.matshow(
        covmat,
        cmap='RdBu',
        norm=colors.SymLogNorm(
            linthresh=0.001,
            linscale=0.5,
            vmin=-1,
            vmax=+1,
        ),
    )
    cbar = fig.colorbar(matrixplot, fraction=0.046, pad=0.04)
    cbar.set_label(label="% of data", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=25)

    # Construct the ticks, labels, and dashed lines
    if specs is not None:
        plt.xticks(specs["tick_locs"], specs["tick_labels"], fontsize=18)
        plt.gca().xaxis.tick_bottom()
        plt.yticks(specs["tick_locs"], specs["tick_labels"], fontsize=18)
        startlocs_lines = [x - 0.5 for x in specs["separation_lines"]]
        ax.vlines(startlocs_lines, -0.5, covmat.shape[0] - 0.5, ls="dashed")
        ax.hlines(startlocs_lines, -0.5, covmat.shape[0] - 0.5, ls="dashed")
        ax.margins(x=0, y=0)

    return fig


def save_heatmap(
    covmat: np.ndarray,
    individual_data: bool,
    figname: pathlib.Path,
    title: str,
    tick_specs: Optional[dict] = None,
):
    fig = heatmap(covmat, title, specs=tick_specs)
    fig.savefig(figname, bbox_inches="tight", dpi=350)

    if not individual_data:
        _logger.info(
            "Plotted covariance/correlation matrix of requested datasets,"
            f" in '{figname.absolute().relative_to(pathlib.Path.cwd())}'"
        )


def main(
    data: list[pathlib.Path],
    destination: pathlib.Path,
    inverse: bool = False,
    norm: bool = True,
    individual_data: bool = False,
    cuts: Optional[dict[str, dict[str, float]]] = None,
):
    """Run covmat plot generation."""
    utils.mkdest(destination)

    normsuf = "" if not norm else "-norm"
    invsuf = "" if not inverse else "-inv"

    covmats = {}
    for ds in data:
        name, datapath = utils.split_data_path(ds)

        covmat = compute(name, datapath, inverse=inverse, norm=norm, cuts=cuts)
        covmats[name] = covmat

        if individual_data:
            figname = destination / f"{name}{normsuf}{invsuf}.pdf"
            save_heatmap(covmat, individual_data, figname, "Cov. Matrix")

            normtag = "normalized " if norm else ""
            invtag = "inverse " if inverse else ""
            _logger.info(
                f"Plotted [b magenta]{normtag}{invtag}[/]covariance matrix"
                f" {covmat.shape} of '{name}',"
                f" in '{figname.absolute().relative_to(pathlib.Path.cwd())}'",
                extra={"markup": True},
            )

    ordered_covmats = order_dict_experiment(covmats)
    ticklabels_specs = construct_ticklabels(ordered_covmats)
    # Plot the total covariance matrix
    totcovmat = scipy.linalg.block_diag(*ordered_covmats.values())
    totcovmat_figname = destination / f"total{normsuf}{invsuf}.pdf"
    save_heatmap(
        covmat=totcovmat,
        individual_data=individual_data,
        figname=totcovmat_figname,
        title=r"$\rm{Covariance~Matrix~-~Experimental~Dataset}$",
        tick_specs=ticklabels_specs,
    )

    # Plot the total correlation coefficient matrix
    totcorrmat = correlation_matrix(totcovmat)
    totcorrmat_figname = destination / f"total_corrmat_{normsuf}{invsuf}.pdf"
    save_heatmap(
        covmat=totcorrmat,
        individual_data=individual_data,
        figname=totcorrmat_figname,
        title=r"$\rm{Correlation~Matrix~-~Experimental~Dataset}$",
        tick_specs=ticklabels_specs,
    )
