import logging
import pathlib
import tempfile
import tarfile

import lhapdf
import pineappl

from . import utils

logger = logging.getLogger(__name__)


def main(grids: pathlib.Path, pdf: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        # extract tar content
        if grids.suffix == ".tar":
            utils.extract_tar(grids, tmpdir)
            grids = tmpdir / "grids"

        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue

            grid = pineappl.grid.Grid.read(gpath)
            pdfset = lhapdf.mkPDF(pdf)

            result = grid.convolute_with_one(2212, pdfset.xfxQ2, pdfset.alphasQ2)
            __import__("pdb").set_trace()
