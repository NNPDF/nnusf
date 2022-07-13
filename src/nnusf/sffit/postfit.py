# -*- coding: utf-8 -*-
import json
import logging
import pathlib
import shutil
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)


def perform_postfit(
    model: pathlib.Path, chi2_threshold: Optional[dict[str, float]]
) -> None:
    if chi2_threshold is not None:
        fitinfos = model.glob("replica_*")
        count_replica_status_fail = 0

        # Create a folder to store the results after postfit
        postfit = model.joinpath("postfit")
        postfit.mkdir(exist_ok=True)

        for nbrep, repinfo in enumerate(fitinfos, start=1):
            with open(f"{repinfo}/fitinfo.json", "r") as file:
                jsonfile = json.load(file)
            tr_chi2 = jsonfile["best_tr_chi2"]
            vl_chi2 = jsonfile["best_vl_chi2"]

            tr_thr = (
                chi2_threshold["tr_max"]
                if "tr_max" in chi2_threshold
                else np.inf
            )
            vl_thr = (
                chi2_threshold["vl_max"]
                if "vl_max" in chi2_threshold
                else np.inf
            )

            if (tr_chi2 <= tr_thr) or (vl_chi2 <= vl_thr):
                repindex = str(repinfo).split("_")[-1]
                dstname = postfit / f"replica_{repindex}"
                # Copy the replica into the postfit folder
                shutil.copytree(repinfo, dstname, dirs_exist_ok=True)
                # Also copies the runcard for the report
                shutil.copy(model.joinpath("runcard.yml"), postfit)
                count_replica_status_fail += 1

        _logger.info(
            f"{nbrep - count_replica_status_fail} replica out"
            f" of the original {nbrep} pass postfit selection."
        )
        _logger.info(
            f"The replicas that passed postfit are stored in: "
            f"'{postfit.absolute().relative_to(pathlib.Path.cwd())}'."
        )
    return


def main(
    model: pathlib.Path, chi2_threshold: Optional[dict[str, float]]
) -> None:
    perform_postfit(model, chi2_threshold)
