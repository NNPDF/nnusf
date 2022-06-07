# -*- coding: utf-8 -*-
import copy

import numpy as np
import yadmark.data.observables

from . import load


def observables() -> dict:
    q2grid, xgrid = load.kin_grids()

    kinematics = np.array(np.meshgrid(q2grid, xgrid)).T.reshape(
        (q2grid.size * xgrid.size, 2)
    )
    kinematics = [
        dict(zip(("x", "Q2", "y"), [float(k) for k in (*kin, 0)])) for kin in kinematics
    ]

    runcard = copy.deepcopy(yadmark.data.observables.default_card)
    runcard = (
        runcard
        | yadmark.data.observables.build(
            ["F2_total", "F3_total"], kinematics=kinematics
        )[0]
    )
    runcard["prDIS"] = "CC"
    runcard["projectile"] = "neutrino"
    #  runcard["interpolation_xgrid"] = xgrid.tolist()

    return {"bodek-yang": runcard}
