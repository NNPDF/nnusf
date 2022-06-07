# -*- coding: utf-8 -*-
import pathlib
import tarfile
import tempfile

import yaml

from .. import utils
from . import bodek_yang, highq


def theory() -> dict:
    runcard = yaml.safe_load(
        (utils.pkg / "theory" / "assets" / "theory_200.yaml").read_text(
            encoding="utf-8"
        )
    )
    return runcard


def main(what: str):
    if what == "by":
        pkg = bodek_yang
    elif what == "hiq":
        pkg = highq
    else:
        raise ValueError(f"Predictions not available for '{what}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        utils.write(theory(), tmpdir / "theory.yaml")
        for name, observable in pkg.runcards.observables().items():
            utils.write(observable, tmpdir / f"obs-{name}.yaml")

        with tarfile.open(pathlib.Path.cwd() / "runcards.tar", "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), arcname="runcards/" + path.name)
