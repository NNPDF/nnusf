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

W2CUT = 3.5
Q2MAX = 25


def plot(
    groups: dict[str, list[list[float]]],
    wcut: bool = True,
    q2cut: bool = True,
    xlog: bool = True,
    ylog: bool = True,
    alpha: bool = True,
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
        shading = (1 - np.tanh(3 * size / total)) if alpha else 1
        if size != 0:
            ax.scatter(
                *kins,
                label=name,
                s=100 / np.power(size, 1 / 4),
                marker=marker,
                alpha=shading,
            )
        else:
            _logger.warn(f"No point received in {name}")

    min_xvalue, max_xvalue = ax.get_xlim()

    if xlog:
        ax.set_xscale("log")
        min_xvalue, max_xvalue = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(min_xvalue, max_xvalue + 0.1, 0.2))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%0.1f$"))

    if ylog:
        ax.set_yscale("log")
    if wcut:
        xvalue = np.arange(min_xvalue, max_xvalue, 5e-2)
        fq2 = lambda x: x * (W2CUT - 0.95) / (1 - x)
        ax.plot(xvalue, fq2(xvalue), color="grey", lw=0.25, zorder=0)
        ax.fill_between(
            xvalue,
            fq2(xvalue),
            fq2(xvalue).min(),
            color="grey",
            alpha=0.15,
            zorder=0,
        )
        ax.text(0.2, 0.1, r"$W^2 \leq 3.5~\mathrm{GeV}^2$", fontsize=15)
    if q2cut:
        xvalue = np.arange(min_xvalue, max_xvalue, 5e-2)
        yvalue = np.repeat(Q2MAX, xvalue.size)
        ytopvl = np.repeat(ax.get_ylim()[-1], xvalue.size)
        ax.plot(xvalue, yvalue, color="#E4B4C2", lw=0.25, zorder=0)
        ax.fill_between(
            xvalue,
            yvalue,
            ytopvl,
            color="#E4B4C2",
            alpha=0.2,
            zorder=0,
        )
        ax.text(0.02, 100, r"$Q^2 \geq 30~\mathrm{GeV}^2$", fontsize=15)

    ax.margins(0.0)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$Q^2~[\rm{GeV}^2]$")
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        mode="expand",
        borderaxespad=0.0,
        ncol=3,
        fontsize=10,
        fancybox=False,
        edgecolor="inherit",
    )
    plt.tight_layout()

    return fig


def main(
    data: list[pathlib.Path],
    destination: pathlib.Path,
    grouping: str = "exp",
    xlog: bool = True,
    ylog: bool = True,
    wcut: bool = True,
    q2cut: bool = True,
    alpha: bool = True,
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

    fig = plot(
        kingroups, wcut=wcut, q2cut=q2cut, xlog=xlog, ylog=ylog, alpha=alpha
    )
    figname = destination / "kinematics.pdf"
    fig.savefig(figname, dpi=350)

    _logger.info(
        "Plotted [b magenta]kinematics[/] of requested datasets,"
        f" in '{figname.absolute().relative_to(pathlib.Path.cwd())}'",
        extra={"markup": True},
    )
