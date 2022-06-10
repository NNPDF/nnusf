# -*- coding: utf-8 -*-
"""Common tools for plotting and handling related data."""
import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__file__)


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
        mask = np.logical_and(mask, mincut, maxcut)

        ncut = -(mincut - 1).sum() - (maxcut - 1).sum()
        _logger.info(f"Cut {ncut} points, in '{var}'")

    return mask
