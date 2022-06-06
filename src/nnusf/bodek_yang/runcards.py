import copy
import pathlib
import tarfile
import tempfile

import numpy as np
import yadmark.data.observables

from . import load
from .. import runcards, utils


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

    return runcard


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        utils.write(runcards.theory(), tmpdir / "theory.yaml")
        utils.write(observables(), tmpdir / "observables.yaml")

        with tarfile.open(pathlib.Path.cwd() / "runcards.tar", "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), arcname="runcards/" + path.name)
