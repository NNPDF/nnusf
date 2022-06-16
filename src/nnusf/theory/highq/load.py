# -*- coding: utf-8 -*-
"""Common definitions and data load."""
import pathlib

import numpy as np

from ...data import loader

Q2CUT = 5

xgrid = np.geomspace(1e-4, 1, 20)
q2grid = np.geomspace(Q2CUT, 1e4, 20)


def unique(ar: np.ndarray) -> object:
    """Check uniqueness for the value.

    Parameters
    ----------
    ar: np.ndarray
        the array to be checked

    Returns
    -------
    object
        the unique value contained in the array

    Raises
    ------
    ValueError
        if the given array contains more than a single value

    """
    uar = np.unique(ar)
    if uar.size > 1:
        raise ValueError("Only one projectile allowed per dataset")

    return uar[0]


def kinematics(name: str, path: pathlib.Path) -> dict:
    """Load data kinematics in a table.

    Parameters
    ----------
    name: str
        dataset to load
    path: os.PathLike

    Returns
    -------
    dict
        mapping of names to kinematics

    """
    data = loader.Loader(name, path)

    y = data.table["y"].values if "y" in data.table else 0.0
    nuclear_mass = unique(data.table["A"].values)
    proj = unique(data.table["projectile"].values)

    kins = dict(
        x=data.table["x"].values,
        y=y,
        Q2=data.table["Q2"].values,
        obs=data.obs,
        exp=data.exp,
        proj=proj,
        A=nuclear_mass,
    )

    return kins
