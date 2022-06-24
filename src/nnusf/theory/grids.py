# -*- coding: utf-8 -*-
"""Compute DIS grids out of given runcards."""
import logging
import pathlib
import tarfile
import tempfile

import yadism

from .. import utils

_logger = logging.getLogger(__name__)


def main(cards: pathlib.Path, destination: pathlib.Path):
    """Run grids computation.

    Parameters
    ----------
    cards: pathlib.Path
        path to runcards archive
    destination: pathlib.Path

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        # extract tar content
        if cards.suffix == ".tar":
            utils.extract_tar(cards, tmpdir, subdirs=1)
            cards = tmpdir / "runcards"

        grids_dest = tmpdir / "grids"
        grids_dest.mkdir()

        theory = utils.read(cards / "theory.yaml")
        for cpath in cards.iterdir():
            _logger.info(f"Attempt computing '{cpath.name}'")
            if cpath.name.startswith("obs"):
                observables = utils.read(cpath, what="yaml")
                output = yadism.run_yadism(theory, observables)

                for obs in observables["observables"]:
                    res_path = grids_dest / f"{obs}.pineappl.lz4"
                    output.dump_pineappl_to_file(res_path, obs)
                    _logger.info(f"Dumped {res_path.name}")

        # TODO: fix the stored files and names.
        data_name = cpath.name.split("-")[1][:-5]
        if ("NU" in data_name and "DXDY" not in data_name) or "NB" in data_name:
            data_name = data_name[:-3]
        with tarfile.open(destination / f"grids-{data_name}.tar", "w") as tar:
            for path in grids_dest.iterdir():
                tar.add(path.absolute(), path.relative_to(tmpdir))
