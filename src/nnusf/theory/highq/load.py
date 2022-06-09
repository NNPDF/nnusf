# -*- coding: utf-8 -*-
"""Common definitions and data load."""
import pathlib

import numpy as np
import pandas as pd

from ...data import loader

sfmap = dict(F2="F2_total", FL="FL_total", F3="F3_total")
projectiles = dict(NU="neutrino", NB="antineutrino")
xsmap = dict(CHORUS="XSCHORUSCC")

Q2CUT = 5

xgrid = np.geomspace(1e-4, 1, 20)
q2grid = np.geomspace(Q2CUT, 1e4, 20)


def kinematics(name: str, path: pathlib.Path) -> pd.DataFrame:
    """Load data kinematics in a table.

    Parameters
    ----------
    name: str
        dataset to load
    path: os.PathLike

    Returns
    -------
    pd.DataFrame
        table with loaded kinematics

    """
    data = loader.Loader(name, path)

    kins = dict(
        x=data.table["x"].values,
        y=data.table["y"].values,
        Q2=data.table["Q2"].values,
        obs=np.random.choice(list(sfmap.keys()) + ["XS"], size=n),
        proj=np.random.randint(-1, 2, size=n),
        A=np.random.randint(1, 100, size=n),
    )

    return pd.DataFrame(kins)
