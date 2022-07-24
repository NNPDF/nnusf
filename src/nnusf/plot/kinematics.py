# -*- coding: utf-8 -*-
"""Generate heatmap plots for covariance matrices."""
import logging
import pathlib
from typing import Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .. import utils
from ..data import loader
from . import utils as putils

_logger = logging.getLogger(__file__)
PARRENT_PATH = pathlib.Path(__file__).parents[1]
MPLSTYLE = PARRENT_PATH.joinpath("plotstyle.mplstyle")
plt.style.use(MPLSTYLE)


def plot(
    groups: dict[str, list[list[float]]],
    wcut: bool = True,
    xlog: bool = True,
    ylog: bool = True,
) -> matplotlib.figure.Figure:
    """Plot (x, Q2) kinematics.

    Parameters
    ----------
    groups: dict
        kinematics data grouped
    ylog: bool
        set logarithmic scale on y axis

    """
    fig, ax = plt.subplots()

    total = sum(len(kins[0]) for kins in groups.values())

    for (name, kins), marker in zip(groups.items(), putils.MARKERS):
        size = len(kins[0])
        if size != 0:
            ax.scatter(
                *kins,
                label=name,
                s=100 / np.power(size, 1 / 4),
                marker=marker,
                alpha=1 - np.tanh(3 * size / total),
            )
        else:
            _logger.warn(f"No point received in {name}")

    if xlog:
        ax.set_xscale("log")
        min_value, max_value = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(min_value, max_value + 0.1, 0.2))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%0.1f$"))
    if ylog:
        ax.set_yscale("log")
    if wcut:
        min_value, max_value = ax.get_xlim()
        xvalue = np.arange(min_value, max_value, 5e-2)
        fq2 = lambda x: x * (3.5 - 0.95) / (1 - x)
        ax.plot(xvalue, fq2(xvalue), ls="dashed", lw=2)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$Q^2~[\rm{GeV}^2]$")
    plt.legend(ncol=2)
    plt.tight_layout()

    return fig


def main(
    data: list[pathlib.Path],
    destination: pathlib.Path,
    grouping: str = "exp",
    xlog: bool = True,
    ylog: bool = True,
    wcut: bool = True,
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

    fig = plot(kingroups, wcut=wcut, xlog=xlog, ylog=ylog)
    figname = destination / "kinematics.pdf"
    fig.savefig(figname)

    _logger.info(
        "Plotted [b magenta]kinematics[/] of requested datasets,"
        f" in '{figname.absolute().relative_to(pathlib.Path.cwd())}'",
        extra={"markup": True},
    )
