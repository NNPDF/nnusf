import copy

import numpy as np
import yaml
import yadmark.data.observables

from . import load, utils


def theory() -> dict:
    runcard = yaml.safe_load((utils.pkg / "theory_200.yaml").read_text())
    return runcard


def observables() -> dict:
    genie = load.load()

    xgrid = genie["xlist"]
    q2grid = genie["q2list"]

    kinematics = np.array(np.meshgrid(xgrid, q2grid)).T.reshape(
        (xgrid.size * q2grid.size, 2)
    )
    kinematics = [dict(zip(("x", "Q2", "y"), (*kin, 0))) for kin in kinematics]

    runcard = copy.deepcopy(yadmark.data.observables.default_card)
    runcard = (
        runcard
        | yadmark.data.observables.build(
            ["F2_total", "F3_total"], kinematics=kinematics
        )[0]
    )
    runcard["prDIS"] = "CC"

    return runcard
