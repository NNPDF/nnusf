import pathlib
import tarfile
import tempfile

from . import load
from .. import runcards, utils


def observables() -> dict:
    kins = load.kinematics()
    __import__("pdb").set_trace()
    return {}


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        utils.write(runcards.theory(), tmpdir / "theory.yaml")
        utils.write(observables(), tmpdir / "observables.yaml")

        with tarfile.open(pathlib.Path.cwd() / "runcards.tar", "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), arcname="runcards/" + path.name)
