# -*- coding: utf-8 -*-
"""Generate runcards for large Q2 yadism predictions.

There are actually two sections:

    - **overlap**: is the large Q2 part that coincide with data from various datasets
    - **boundary**: is the further boundary condition, on regular grids in x
        and Q2 and directly on structure functions

"""
import copy
import logging
import pathlib
from typing import Optional

import yadmark.data.observables

from . import load

_logger = logging.getLogger(__name__)


def overlap(datasets: list[str], path: pathlib.Path) -> dict:
    """Provide data overlap yadism runcards.

    Returns
    -------
    dict
        id to observables runcard mapping

    """
    kins = load.kinematics(datasets[0], path)

    run_nu = copy.deepcopy(yadmark.data.observables.default_card)
    run_nu["prDIS"] = "CC"
    run_nu["ProjectileDIS"] = "neutrino"
    nu = kins["proj"] >= 0

    run_nb = copy.deepcopy(run_nu)
    run_nb["ProjectileDIS"] = "antineutrino"
    nb = kins["proj"] <= 0

    for proj, run in [(nu, run_nu), (nb, run_nb)]:
        for obs, obsname in load.sfmap.items():
            obsfilter = kins["obs"] == obs
            obskins = list(kins[proj][obsfilter][["x", "Q2", "y"]].T.to_dict().values())
            run["observables"][obsname] = obskins

        #  xsfilter = kins["obs"] == "XS"
        # TODO: recognize experiment, plug the appropriate XS for each
        # experiment

    run_nu_extra = copy.deepcopy(run_nu)
    run_nu_extra["observables"] = {}

    run_nb_extra = copy.deepcopy(run_nb)
    run_nb_extra["observables"] = {}

    for sfname in load.sfmap.values():
        run_nu_extra["observables"][sfname] = []
        run_nb_extra["observables"][sfname] = []
        for x in load.xgrid:
            for q2 in load.q2grid:
                run_nu_extra["observables"][sfname].append(dict(x=x, Q2=q2, y=0))
                run_nb_extra["observables"][sfname].append(dict(x=x, Q2=q2, y=0))

    _logger.info("Data overlapping runcards generated.")
    return dict(nu=run_nu, nb=run_nb, nu_extra=run_nb_extra, nb_extra=run_nb_extra)


def boundary() -> dict:
    """Provide boundary conditions enforcing yadism runcards.

    Returns
    -------
    dict
        id to observables runcard mapping

    """
    _logger.info("Large Q2 boundary conditions runcard generated.")
    return dict()


def observables(datasets: list[str], path: Optional[pathlib.Path]) -> dict:
    """Collect all yadism runcards for large Q2 region.

    Returns
    -------
    dict
        id to observables runcard mapping

    """
    if len(datasets) == 0 or path is None:
        if path is None:
            _logger.warning("No path to data folder provided.")
        else:
            _logger.warning("No data requested.")

        _logger.warning("No data overlapping runcard will be generated.")
        return boundary()

    return overlap(datasets, path) | boundary()
