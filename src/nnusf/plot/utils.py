# -*- coding: utf-8 -*-
"""Common tools for plotting and handling related data."""
import logging

import matplotlib.colors as clr
import numpy as np
import pandas as pd

from ..data import loader

_logger = logging.getLogger(__file__)

MARKERS = ["o", "s", "D", "*", "^", ">", "X"]


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
    kins = {k: table[k] for k in ["x", "y", "Q2"] if k in table}
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
        else:
            raise ValueError

        if label not in groups:
            groups[label] = [lds]
        else:
            groups[label].append(lds)

    return groups
