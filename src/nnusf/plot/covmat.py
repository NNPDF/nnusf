# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .. import utils
from ..data import loader

_logger = logging.getLogger(__file__)


def heatmap(
    name: str, datapath: pathlib.Path, inverse: bool = False
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

    Returns
    -------
    matplotlib.figure.Figure
        plotted figure
    np.ndarray
        (inverse) covariance matrix computed

    """
    data = loader.Loader(name, datapath)

    covmat = data.covariance_matrix
    if inverse:
        covmat = np.linalg.inv(covmat)

    fig = plt.figure()
    sns.heatmap(covmat)

    return fig, covmat


def main(data: list[pathlib.Path], destination: pathlib.Path, inverse: bool = False):
    """Run covmat plot generation."""
    utils.mkdest(destination)

    covmats = {}
    for ds in data:
        name = ds.stem.strip("DATA_")

        fig, covmat = heatmap(name, ds.parents[1], inverse=inverse)
        inv = "" if not inverse else "-inv"
        figname = destination / f"{name}{inv}.png"
        fig.savefig(figname)
        _logger.info(
            f"Plotted covariance matrix of '{name}' in '{figname.relative_to(pathlib.Path.cwd())}'"
        )

        covmats[name] = covmat
