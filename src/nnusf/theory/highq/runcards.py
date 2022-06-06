import copy
import pathlib
import tarfile
import tempfile

import yadmark.data.observables

from . import load
from .. import runcards
from ... import utils


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


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        utils.write(runcards.theory(), tmpdir / "theory.yaml")
        for name, observable in observables().items():
            utils.write(observable, tmpdir / f"obs-{name}.yaml")

        with tarfile.open(pathlib.Path.cwd() / "runcards.tar", "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), arcname="runcards/" + path.name)
