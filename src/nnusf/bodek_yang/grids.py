import logging
import pathlib
import tempfile
import tarfile

import yadism

from .. import utils

logger = logging.getLogger(__name__)


def main(cards: pathlib.Path):
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
            if cpath.name.startswith("obs"):
                observables = utils.read(cpath)
                output = yadism.run_yadism(theory, observables)

                for obs in observables["observables"]:
                    res_path = grids_dest / f"{obs}.pineappl.lz4"
                    output.dump_pineappl_to_file(res_path, obs)
                    logger.info(f"Dumped {res_path.name}")

        with tarfile.open(pathlib.Path.cwd() / "grids.tar", "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), path.relative_to(tmpdir))
