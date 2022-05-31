import pathlib
import tempfile
import tarfile

import yadism

from . import utils


def main(cards_path: pathlib.Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        # extract tar content
        if cards_path.suffix == ".tar":
            with tarfile.open(cards_path) as tar:
                tar.extractall(tmpdir)

            content = iter(tmpdir.iterdir())
            cards_path = next(content)
            try:
                next(content)
                raise ValueError(
                    "A single folder is supposed to be contained by the tar file,"
                    " but more files have been detected"
                )
            except StopIteration:
                pass

        theory = utils.read(cards_path / "theory.yaml")
        for cpath in cards_path.iterdir():
            if cpath.name.startswith("obs"):
                observables = utils.read(cpath)
                output = yadism.run_yadism(theory, observables)

                __import__("pdb").set_trace()
