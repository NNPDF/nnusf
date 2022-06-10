# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib
from typing import Optional

import matplotlib.colors as clr
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import seaborn as sns

from .. import utils
from ..data import loader
from . import utils as putils

_logger = logging.getLogger(__file__)


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


def heatmap(covmat: np.ndarray, symlog: bool = False) -> matplotlib.figure.Figure:
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
    extra = {}
    if symlog:
        extra["norm"] = clr.SymLogNorm(abs(covmat.min()))
        _logger.info("Symmetric log scale enabled.")

    fig = plt.figure()
    sns.heatmap(covmat, **extra)

    return fig


def main(
    data: list[pathlib.Path],
    destination: pathlib.Path,
    inverse: bool = False,
    norm: bool = True,
    symlog: bool = False,
    cuts: Optional[dict[str, dict[str, float]]] = None,
):
    """Run covmat plot generation."""
    utils.mkdest(destination)

    normsuf = "" if not norm else "-norm"
    invsuf = "" if not inverse else "-inv"

    covmats = {}
    for ds in data:
        name = ds.stem.strip("DATA_")

        covmat = compute(name, ds.parents[1], inverse=inverse, norm=norm, cuts=cuts)
        covmats[name] = covmat

        fig = heatmap(covmat, symlog=symlog)
        figname = destination / f"{name}{normsuf}{invsuf}.png"
        fig.savefig(figname)

        normtag = "normalized " if norm else ""
        invtag = "inverse " if inverse else ""
        _logger.info(
            f"Plotted [b magenta]{normtag}{invtag}[/]covariance matrix"
            f" {covmat.shape} of '{name}',"
            f" in '{figname.relative_to(pathlib.Path.cwd())}'",
            extra={"markup": True},
        )

    totcovmat = scipy.linalg.block_diag(*covmats.values())

    fig = heatmap(totcovmat, symlog=symlog)
    figname = destination / f"total{normsuf}{invsuf}.png"
    fig.savefig(figname)

    _logger.info(
        "Plotted covariance matrix of requested datasets,"
        f" in '{figname.relative_to(pathlib.Path.cwd())}'"
    )
