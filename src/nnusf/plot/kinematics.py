# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from .. import utils
from ..data import loader
from . import utils as putils

_logger = logging.getLogger(__file__)


def main(data: list[pathlib.Path], destination: pathlib.Path, grouping: str = "exp"):
    """Run kinematic plot generation."""
    utils.mkdest(destination)

    fig = plt.figure()

    groups = {}
    total = 0
    for ds in data:
        name, datapath = utils.split_data_path(ds)
        lds = loader.Loader(name, datapath)
        total += lds.n_data

        kins = [lds.table[k].values.tolist() for k in ["x", "Q2"]]

        if grouping == "exp":
            label = lds.exp
        else:
            raise ValueError

        if label not in groups:
            groups[label] = [[], []]

        for i in range(2):
            groups[label][i].extend(kins[i])

    for (name, kins), marker in zip(groups.items(), putils.MARKERS):
        size = len(kins[0])
        plt.scatter(
            *kins,
            label=name,
            s=100 / np.power(size, 1 / 3),
            marker=marker,
            alpha=1 - np.tanh(2 * size / total)
        )

    plt.yscale("log")
    plt.xlabel("$x$")
    plt.ylabel("$Q^2$")
    plt.legend()
    plt.tight_layout()

    figname = destination / "kinematics.png"
    fig.savefig(figname)

    _logger.info("kinematic plot")
