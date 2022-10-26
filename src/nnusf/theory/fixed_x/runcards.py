# -*- coding: utf-8 -*-
"""Generate runcards with x and A fixed."""
import numpy as np

import copy 
import yadmark 
from .. import defs


q2_grid = np.geomspace(5**2, 10**6, 400)

def observables(x: float, A: int) -> dict:
    """Collect all yadism runcards."""
    run_nu = copy.deepcopy(yadmark.data.observables.default_card)
    run_nu["prDIS"] = "CC"
    run_nu["ProjectileDIS"] = "neutrino"

    kins = []
    for q2 in q2_grid:
        kins.append({"x": float(x), "Q2": float(q2), "y": 0.0})
    run_nu["observables"] = {
        "F2": kins, "F3": kins, "FL": kins
    }
    run_nu["TargetDIS"] = defs.targets[A]
    run_nb = copy.deepcopy(run_nu)
    run_nb["ProjectileDIS"] = "antineutrino"

    return {f"nu_A_{A}": run_nu, f"nub_A_{A}": run_nb}
