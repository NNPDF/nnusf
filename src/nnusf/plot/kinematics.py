# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib

import matplotlib.pyplot as plt

from .. import utils
from ..data import loader

_logger = logging.getLogger(__file__)


def main(data: list[pathlib.Path], destination: pathlib.Path):
    """Run kinematic plot generation."""
    utils.mkdest(destination)

    fig = plt.figure()

    for ds in data:
        name, datapath = utils.split_data_path(ds)

        lds = loader.Loader(name, datapath)
        kins = [lds.table[k].values for k in ["x", "Q2"]]
        plt.scatter(*kins, label=name, s=10)

    plt.legend()

    figname = destination / "kinematics.png"
    fig.savefig(figname)

    _logger.info("kinematic plot")
