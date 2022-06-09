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

import pandas as pd
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
    runcards = {}

    run_nu = copy.deepcopy(yadmark.data.observables.default_card)
    run_nu["prDIS"] = "CC"
    run_nu["ProjectileDIS"] = "neutrino"

    run_nb = copy.deepcopy(run_nu)
    run_nb["ProjectileDIS"] = "antineutrino"

    for dataset in datasets:
        kins = load.kinematics(dataset, path)

        ds_cards = {}
        if kins["proj"] > 0:
            ds_cards[dataset] = copy.deepcopy(run_nu)
        elif kins["proj"] < 0:
            ds_cards[dataset] = copy.deepcopy(run_nb)
        else:
            ds_cards[f"{dataset}_NU"] = copy.deepcopy(run_nu)
            ds_cards[f"{dataset}_NB"] = copy.deepcopy(run_nb)

        for run in ds_cards.values():
            obsname = load.sfmap[kins["obs"]]
            obskins = list(
                pd.DataFrame({k: v for k, v in kins.items() if k in ["x", "Q2", "y"]})
                .T.to_dict()
                .values()
            )
            run["observables"][obsname] = obskins

        runcards |= ds_cards

    _logger.info("Data overlapping runcards generated.")
    return runcards


def boundary() -> dict:
    """Provide boundary conditions enforcing yadism runcards.

    Returns
    -------
    dict
        id to observables runcard mapping

    """
    run_nu_extra = copy.deepcopy(yadmark.data.observables.default_card)
    run_nu_extra["prDIS"] = "CC"
    run_nu_extra["ProjectileDIS"] = "neutrino"
    run_nu_extra["observables"] = {}

    run_nb_extra = copy.deepcopy(run_nu_extra)
    run_nu_extra["ProjectileDIS"] = "antineutrino"
    run_nb_extra["observables"] = {}

    for sfname in load.sfmap.values():
        run_nu_extra["observables"][sfname] = []
        run_nb_extra["observables"][sfname] = []
        for x in load.xgrid:
            for q2 in load.q2grid:
                run_nu_extra["observables"][sfname].append(dict(x=x, Q2=q2, y=0))
                run_nb_extra["observables"][sfname].append(dict(x=x, Q2=q2, y=0))

    _logger.info("Large Q2 boundary conditions runcard generated.")
    return dict(nu_extra=run_nb_extra, nb_extra=run_nb_extra)


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
