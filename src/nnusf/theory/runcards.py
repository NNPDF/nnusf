# -*- coding: utf-8 -*-
"""Utilities and runners to generate yadism runcards."""
import logging
import pathlib
import tarfile
import tempfile
from typing import Optional

import numpy as np
import yaml

from .. import utils
from ..data import loader
from . import bodek_yang, data_vs_theory, defs, highq, yadknots

_logger = logging.getLogger(__name__)


def theory(update: Optional[dict] = None) -> dict:
    """Load and return internal theory runcard."""
    runcard = yaml.safe_load(
        (utils.pkg / "theory" / "assets" / "theory_nnusf.yaml").read_text(
            encoding="utf-8"
        )
    )

    if update is not None:
        runcard.update(update)
        _logger.info(f"Base theory updated with {update}")

    return runcard


def dump(
    theory_card: dict,
    observables_cards: dict[str, dict],
    destination: pathlib.Path = pathlib.Path.cwd(),
):
    """Dump runcards to tar in a given destination.

    Raises
    ------
    NotADirectoryError
        if destination is given, but it is not a directory

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        utils.write(theory_card, tmpdir / "theory.yaml")
        for name, observable in observables_cards.items():
            utils.write(observable, tmpdir / f"obs-{name}.yaml")

        utils.mkdest(destination)

        tarpath = destination / "runcards.tar"
        with tarfile.open(tarpath, "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), arcname="runcards/" + path.name)

        _logger.info(
            f"Runcards have been dumped to '{tarpath.relative_to(pathlib.Path.cwd())}'"
        )


def by(
    theory_update: Optional[dict],
    obs_update: Optional[dict],
    destination: pathlib.Path,
):
    """Generate Bodek-Yang yadism runcards."""
    obs = bodek_yang.runcards.observables()
    if obs_update is not None:
        # cryptic code: update `v` and return it, since `v.update` returns `None`
        obs = {k: v.update(obs_update) or v for k, v in obs.items()}
        _logger.info(f"Base theory updated with {obs_update}")

    dump(
        theory(theory_update),
        obs,
        destination=destination,
    )


def datasets_path(datasets) -> pathlib.Path:
    path = None
    if len(datasets) > 0:
        path = datasets[0].parents[1].absolute()
        for ds in datasets:
            assert ds.parents[1].absolute() == path
    else:
        raise ValueError(
            "No datasets passed, impossible to infer datasets location."
        )

    return path


def hiq(datasets: list[pathlib.Path], destination: pathlib.Path):
    """Generate large Q2 yadism runcards."""
    path = datasets_path(datasets)
    dump(
        theory(),
        highq.runcards.observables(
            ["_".join(ds.stem.split("_")[1:]) for ds in datasets], path
        ),
        destination=destination,
    )


def update_theory(
    name: str, path: pathlib.Path, activate_scale_var: bool
) -> list:
    """Update theory runcard"""
    data = loader.Loader(name, path)
    ths = []
    if activate_scale_var:
        # 7 points prescription (xir, xif)
        for xif, xir in defs.seven_points:
            ths.append(
                theory(
                    {
                        "MP": float(np.unique(data.table["m_nucleon"])),
                        "XIF": xif,
                        "XIR": xir,
                    }
                )
            )
    else:
        ths.append(theory({"MP": float(np.unique(data.table["m_nucleon"]))}))
    return ths


def dvst(
    datasets: list[pathlib.Path],
    destination: pathlib.Path,
    activate_scale_var: bool,
):
    """Generate yadism runcards for all datapoints."""
    path = datasets_path(datasets)
    utils.mkdest(destination)
    # Loop on datasets
    for dataset in datasets:
        data_name = "_".join(dataset.stem.split("_")[1:])
        ocards = data_vs_theory.runcards.observables([data_name], path)
        # loop on theories
        theory_cards = update_theory(data_name, path, activate_scale_var)
        for theory_card in theory_cards:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = pathlib.Path(tmpdir)
                utils.write(theory_card, tmpdir / "theory.yaml")
                for name, observable_card in ocards.items():
                    utils.write(observable_card, tmpdir / f"obs-{name}.yaml")

                tarpath = (
                    destination
                    / f"runcards-{data_name}_xif{theory_card['XIF']}_xir{theory_card['XIR']}.tar"
                )
                with tarfile.open(tarpath, "w") as tar:
                    for tmppath in tmpdir.iterdir():
                        tar.add(
                            tmppath.absolute(),
                            arcname="runcards/" + tmppath.name,
                        )

                _logger.info(
                    f"Runcards have been dumped to '{tarpath.relative_to(pathlib.Path.cwd())}'"
                )


def yknots(x: dict, q2: dict, A: int, destination: pathlib.Path):
    """Generate yadism runcards with the given kinametics"""
    utils.mkdest(destination)
    ocards = yadknots.runcards.observables(x, q2, A)
    theory_card = theory()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        utils.write(theory_card, tmpdir / "theory.yaml")
        for name, observable_card in ocards.items():
            utils.write(observable_card, tmpdir / f"obs-{name}.yaml")

            tarpath = destination / f"runcards-yadknots_A{A}.tar"
            with tarfile.open(tarpath, "w") as tar:
                for tmppath in tmpdir.iterdir():
                    tar.add(
                        tmppath.absolute(), arcname="runcards/" + tmppath.name
                    )

            _logger.info(
                f"Runcards have been dumped to '{tarpath.relative_to(pathlib.Path.cwd())}'"
            )
