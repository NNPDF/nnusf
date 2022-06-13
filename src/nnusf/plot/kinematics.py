# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from .. import utils
from ..data import loader

_logger = logging.getLogger(__file__)


def main(data: list[pathlib.Path], destination: pathlib.Path):
    """Run kinematic plot generation."""
    utils.mkdest(destination)

    fig = plt.figure()

    experiments = {}
    total = 0
    for ds in data:
        name, datapath = utils.split_data_path(ds)

        lds = loader.Loader(name, datapath)
        kins = [lds.table[k].values.tolist() for k in ["x", "Q2"]]

        if lds.exp not in experiments:
            experiments[lds.exp] = [[], []]

        for i in range(2):
            experiments[lds.exp][i].extend(kins[i])

        total += len(kins[0])

    for (exp, kins), marker in zip(
        experiments.items(), ["o", "s", "D", "*", "^", ">", "X"]
    ):
        size = len(kins[0])
        plt.scatter(
            *kins,
            label=exp,
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
