# -*- coding: utf-8 -*-
"""Common tools for plotting and handling related data."""
import logging

import matplotlib.colors as clr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..data import loader

_logger = logging.getLogger(__file__)

MARKERS = ["o", "H", "p", "*", "h", "s", "X"]


def cuts(cuts: dict[str, dict[str, float]], table: pd.DataFrame) -> np.ndarray:
    """Generate a mask from given kinematic cuts.

    Parameters
    ----------
    cuts: dict
        dictionary specifying cuts
    data: pd.DataFrame
        the table containing kinematics variables for data

    Returns
    -------
    np.ndarray
        the mask generated

    """
    kins = {k: table[k] for k in ["x", "Q2", "W2"] if k in table}
    mask = np.full_like(table["x"], True, dtype=np.bool_)

    for var, kin in kins.items():
        if var not in cuts:
            continue

        mink = cuts[var]["min"] if "min" in cuts[var] else -np.inf
        maxk = cuts[var]["max"] if "max" in cuts[var] else np.inf
        mincut = mink < kin.values
        maxcut = kin.values < maxk
        mask = mask & mincut & maxcut

        ncut = (1 - mincut).sum() + (1 - maxcut).sum()
        _logger.info(f"Cut {ncut} points, in '{var}'")

    return mask


def symlog_color_scale(ar: np.ndarray) -> clr.SymLogNorm:
    """Tune symmetric color scale on array.

    Parameters
    ----------
    ar: np.ndarray
        array to fit the scale on

    Returns
    -------
    clr.SymLogNorm
        matplotlib color specification generated

    """
    c = clr.SymLogNorm(abs(ar[ar != 0.0]).min())
    _logger.info(
        "Symmetric [b magenta]log scale[/] enabled.", extra={"markup": True}
    )
    return c


def group_data(
    data: list[loader.Loader], grouping: str
) -> dict[str, list[loader.Loader]]:
    """Group data by given criterion."""
    groups = {}

    for lds in data:
        if grouping == "exp":
            label = lds.exp
        elif grouping == "dataset":
            label = lds.name
        else:
            raise ValueError

        if label not in groups:
            groups[label] = [lds]
        else:
            groups[label].append(lds)

    return groups


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
