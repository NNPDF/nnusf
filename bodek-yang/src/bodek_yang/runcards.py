import copy
import pathlib
import tarfile
import tempfile

import numpy as np
import yaml
import yadmark.data.observables

from . import load, utils

Q2MIN = 0.5**2
Q2MAX = 5**2
XMIN = 1e-3


def theory() -> dict:
    runcard = yaml.safe_load(
        (utils.pkg / "theory_200.yaml").read_text(encoding="utf-8")
    )
    return runcard


def observables() -> dict:
    genie = load.load()

    xgrid = np.array(list(filter(lambda x: XMIN < x, genie["xlist"])))
    q2grid = np.array(list(filter(lambda q2: Q2MIN < q2 < Q2MAX, genie["q2list"])))

    print(f"x: #{xgrid.size} {xgrid.min():4.3e} - {xgrid.max()}")
    print(
        f"Q: #{q2grid.size} {np.sqrt(q2grid.min()):4.3e} - {np.sqrt(q2grid.max()):4.3e}"
    )

    kinematics = np.array(np.meshgrid(xgrid, q2grid)).T.reshape(
        (xgrid.size * q2grid.size, 2)
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
    #  runcard["interpolation_xgrid"] = xgrid.tolist()

    return runcard


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        utils.write(theory(), tmpdir / "theory.yaml")
        utils.write(observables(), tmpdir / "observables.yaml")

        with tarfile.open(pathlib.Path.cwd() / "runcards.tar", "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), arcname="runcards/" + path.name)
