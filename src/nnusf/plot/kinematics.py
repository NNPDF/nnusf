# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from .. import utils
from ..data import loader
from . import utils as putils

_logger = logging.getLogger(__file__)


def plot(
    groups: dict[str, list[list[float]]], ylog: bool = True
) -> matplotlib.figure.Figure:
    """Plot (x, Q2) kinematics.

    Parameters
    ----------
    groups: dict
        kinematics data grouped
    ylog: bool
        set logarithmic scale on y axis

    """
    fig = plt.figure()

    total = sum(len(kins[0]) for kins in groups.values())

    for (name, kins), marker in zip(groups.items(), putils.MARKERS):
        size = len(kins[0])
        if size != 0:
            plt.scatter(
                *kins,
                label=name,
                s=100 / np.power(size, 1 / 3),
                marker=marker,
                alpha=1 - np.tanh(2 * size / total),
            )
        else:
            _logger.warn(f"No point received in {name}")

    if ylog:
        plt.yscale("log")
    plt.xlabel("$x$")
    plt.ylabel("$Q^2$")
    plt.legend()
    plt.tight_layout()

    return fig


def main(
    data: list[pathlib.Path],
    destination: pathlib.Path,
    grouping: str = "exp",
    ylog: bool = True,
    cuts: Optional[dict[str, dict[str, float]]] = None,
):
    """Run kinematic plot generation."""
    utils.mkdest(destination)

    groups = putils.group_data(
        [loader.Loader(*utils.split_data_path(ds)) for ds in data],
        grouping=grouping,
    )

    kingroups = {}
    for name, grp in groups.items():
        kingroups[name] = []
        for k in ("x", "Q2"):
            kins = []
            for lds in grp:
                values = lds.table[k].values
                if cuts is not None:
                    mask = putils.cuts(cuts, lds.table)
                    values = values[mask]

                kins.extend(values.tolist())

            kingroups[name].append(kins)

    fig = plot(kingroups, ylog=ylog)
    figname = destination / "kinematics.png"
    fig.savefig(figname)

    _logger.info(
        "Plotted [b magenta]kinematics[/] of requested datasets,"
        f" in '{figname.relative_to(pathlib.Path.cwd())}'",
        extra={"markup": True},
    )
