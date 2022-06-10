# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib

from .. import utils
from ..data import loader

_logger = logging.getLogger(__file__)


def main(data: list[pathlib.Path], destination: pathlib.Path):
    """Run kinematic plot generation."""
    utils.mkdest(destination)
    for ds in data:
        name, datapath = utils.split_data_path(ds)

        lds = loader.Loader(name, datapath)
        kins = {k: lds.table[k] for k in ["x", "y", "Q2"] if k in lds.table}

    _logger.info("kinematic plot")
