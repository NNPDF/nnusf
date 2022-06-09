# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib
from typing import Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import seaborn as sns

from .. import utils
from ..data import loader

_logger = logging.getLogger(__file__)


def heatmap(
    name: str,
    datapath: pathlib.Path,
    inverse: bool = False,
    cuts: Optional[dict[str, dict[str, float]]] = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Compute covmat and plot its heatmap.

    Parameters
    ----------
    name: str
        name of the requested dataset
    datapath: pathlib.Path
        path to commondata
    inverse: bool
        if `True`, compute and plot the inverse of the covariance matrix
        (default: `False`)
    cuts: dict
        kinematic cuts

    Returns
    -------
    matplotlib.figure.Figure
        plotted figure
    np.ndarray
        (inverse) covariance matrix computed

    """
    data = loader.Loader(name, datapath)
    covmat = data.covariance_matrix

    if cuts is not None:
        kins = {k: data.table[k] for k in ["x", "y", "Q2"] if k in data.table}
        mask = np.full_like(data.table["x"], True, dtype=np.bool_)
        for var, kin in kins.items():
            if var not in cuts:
                continue

            mink = cuts[var]["min"] if "min" in cuts[var] else -np.inf
            maxk = cuts[var]["max"] if "max" in cuts[var] else np.inf
            mincut = mink < kin.values
            maxcut = kin.values < maxk
            mask = np.logical_and(mask, mincut, maxcut)

            ncut = -(mincut - 1).sum() - (maxcut - 1).sum()
            _logger.info(f"Cut {ncut} points, in '{var}'")

        covmat = covmat[mask][:, mask]

    if inverse:
        covmat = np.linalg.inv(covmat)

    fig = plt.figure()
    sns.heatmap(covmat)

    return fig, covmat


def main(
    data: list[pathlib.Path],
    destination: pathlib.Path,
    inverse: bool = False,
    cuts: Optional[dict[str, dict[str, float]]] = None,
):
    """Run covmat plot generation."""
    utils.mkdest(destination)

    inv = "" if not inverse else "-inv"

    covmats = {}
    for ds in data:
        name = ds.stem.strip("DATA_")

        fig, covmat = heatmap(name, ds.parents[1], inverse=inverse, cuts=cuts)
        figname = destination / f"{name}{inv}.png"
        fig.savefig(figname)
        _logger.info(
            f"Plotted covariance matrix {covmat.shape} of '{name}',"
            f" in '{figname.relative_to(pathlib.Path.cwd())}'"
        )

        covmats[name] = covmat

    totcovmat = scipy.linalg.block_diag(*covmats.values())

    fig = plt.figure()
    sns.heatmap(totcovmat)

    figname = destination / f"total{inv}.png"
    fig.savefig(figname)

    _logger.info(
        "Plotted covariance matrix of requested datasets,"
        f" in '{figname.relative_to(pathlib.Path.cwd())}'"
    )
