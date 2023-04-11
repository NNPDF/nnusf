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

# The following JLAB_ numbers represent the value
# of the Effective Charge which was taken from the
# paper: https://arxiv.org/pdf/2205.01169.pdf
JLAB = [
    {"Q": 0.143, "a_g1": [3.064, 0.043, 0.018]},
    {"Q": 0.156, "a_g1": [3.129, 0.046, 0.019]},
    {"Q": 0.171, "a_g1": [2.955, 0.046, 0.023]},
    {"Q": 0.187, "a_g1": [3.083, 0.044, 0.024]},
    {"Q": 0.204, "a_g1": [3.022, 0.049, 0.024]},
    {"Q": 0.223, "a_g1": [3.002, 0.052, 0.027]},
    {"Q": 0.243, "a_g1": [2.988, 0.055, 0.031]},
    {"Q": 0.266, "a_g1": [2.947, 0.060, 0.035]},
    {"Q": 0.291, "a_g1": [2.983, 0.065, 0.035]},
    {"Q": 0.317, "a_g1": [2.961, 0.062, 0.038]},
    {"Q": 0.347, "a_g1": [2.730, 0.070, 0.044]},
    {"Q": 0.379, "a_g1": [2.853, 0.077, 0.040]},
    {"Q": 0.414, "a_g1": [2.745, 0.076, 0.041]},
    {"Q": 0.452, "a_g1": [2.779, 0.090, 0.043]},
    {"Q": 0.494, "a_g1": [2.451, 0.094, 0.044]},
    {"Q": 0.540, "a_g1": [2.397, 0.092, 0.039]},
    {"Q": 0.590, "a_g1": [2.349, 0.101, 0.040]},
    {"Q": 0.645, "a_g1": [2.431, 0.109, 0.043]},
    {"Q": 0.704, "a_g1": [1.996, 0.131, 0.104]},
    {"Q": 0.187, "a_g1": [3.016, 0.009, 0.027]},
    {"Q": 0.239, "a_g1": [2.973, 0.015, 0.035]},
    {"Q": 0.281, "a_g1": [2.952, 0.021, 0.041]},
    {"Q": 0.316, "a_g1": [2.929, 0.017, 0.048]},
    {"Q": 0.387, "a_g1": [2.815, 0.021, 0.076]},
    {"Q": 0.447, "a_g1": [2.704, 0.025, 0.086]},
    {"Q": 0.490, "a_g1": [2.575, 0.031, 0.053]},
    {"Q": 0.775, "a_g1": [1.743, 0.007, 0.071]},
    {"Q": 0.835, "a_g1": [1.571, 0.007, 0.101]},
    {"Q": 0.917, "a_g1": [1.419, 0.009, 0.132]},
    {"Q": 0.986, "a_g1": [1.341, 0.010, 0.147]},
    {"Q": 1.088, "a_g1": [1.272, 0.010, 0.156]},
    {"Q": 1.167, "a_g1": [1.121, 0.013, 0.153]},
    {"Q": 1.261, "a_g1": [0.955, 0.016, 0.146]},
    {"Q": 1.384, "a_g1": [0.874, 0.016, 0.269]},
    {"Q": 1.522, "a_g1": [0.730, 0.012, 0.280]},
    {"Q": 1.645, "a_g1": [0.708, 0.009, 0.257]},
    {"Q": 1.795, "a_g1": [0.617, 0.007, 0.254]},
    {"Q": 1.967, "a_g1": [0.581, 0.006, 0.223]},
    {"Q": 2.177, "a_g1": [0.636, 0.003, 0.187]},
]


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


def format_jlab(data: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses the JLAB data and returns the arrays of the central values
    with the corresponding uncertainties added in quadrature.

    Parameters:
    -----------
    data: str
        name of the JLAB dataset

    Returns:
    --------
    tuple(np.ndarray, np.ndarray, np.ndarray):
        tuple containing the Q2 values, central values and uncertainties
    """
    q_values, central, error = [], [], []
    for dic_values in globals()[data]:
        stat = dic_values["a_g1"][1]
        syst = dic_values["a_g1"][2]
        unct = np.sqrt(stat**2 + syst**2)

        q_values.append(dic_values["Q"])
        error.append(unct)
        central.append(dic_values["a_g1"][0])

    return np.asarray(q_values), np.asarray(central), np.asarray(error)
