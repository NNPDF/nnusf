# -*- coding: utf-8 -*-
"""Utilities and runners to generate yadism runcards."""
import logging
import pathlib
import tarfile
import tempfile

import yaml

from .. import utils
from . import bodek_yang, highq

_logger = logging.getLogger(__name__)


def theory() -> dict:
    """Load and return internal theory runcard."""
    runcard = yaml.safe_load(
        (utils.pkg / "theory" / "assets" / "theory_200.yaml").read_text(
            encoding="utf-8"
        )
    )
    return runcard


def dump(
    theory_card: dict,
    observables_cards: dict[str, dict],
    destination: pathlib.Path = pathlib.Path.cwd(),
):
    """Dump runcards to tar in a given destination."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        utils.write(theory_card, tmpdir / "theory.yaml")
        for name, observable in observables_cards.items():
            utils.write(observable, tmpdir / f"obs-{name}.yaml")

        tarpath = destination / "runcards.tar"
        with tarfile.open(tarpath, "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), arcname="runcards/" + path.name)

        _logger.info(
            f"Runcards have been dumped to '{tarpath.relative_to(pathlib.Path.cwd())}'"
        )


def by(destination: pathlib.Path):
    """Generate Bodek-Yang yadism runcards."""
    dump(theory(), bodek_yang.runcards.observables())


def hiq(datasets: list[pathlib.Path], destination: pathlib.Path):
    """Generate large Q2 yadism runcards."""
    path = None
    if len(datasets) > 0:
        path = datasets[0].parents[1].absolute()
        for ds in datasets:
            assert ds.parents[1].absolute() == path

    dump(
        theory(),
        highq.runcards.observables(
            ["_".join(ds.stem.split("_")[1:]) for ds in datasets], path
        ),
    )
