import copy

import yadmark.data.observables

from . import load


def observables() -> dict:
    kins = load.kinematics()

    run_nu = copy.deepcopy(yadmark.data.observables.default_card)
    run_nu["prDIS"] = "CC"
    run_nu["ProjectileDIS"] = "neutrino"
    nu = kins["proj"] >= 0

    run_nb = copy.deepcopy(run_nu)
    run_nb["ProjectileDIS"] = "antineutrino"
    nb = kins["proj"] <= 0

    for proj, run in [(nu, run_nu), (nb, run_nb)]:
        for obs, obsname in load.obsmap.items():
            obsfilter = kins["obs"] == obs
            obskins = list(kins[proj][obsfilter][["x", "Q2", "y"]].T.to_dict().values())
            run["observables"][obsname] = obskins

    return dict(nu=run_nu, nb=run_nb)
