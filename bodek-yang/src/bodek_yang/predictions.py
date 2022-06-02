import logging
import pathlib
import tempfile
import tarfile

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import pineappl

from . import load, utils

logger = logging.getLogger(__name__)


def main(grids: pathlib.Path, pdf: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        # extract tar content
        if grids.suffix == ".tar":
            utils.extract_tar(grids, tmpdir)
            grids = tmpdir / "grids"

        preds_dest = tmpdir / "preds"
        preds_dest.mkdir()

        genie = load.load()
        xgrid, q2grid = np.meshgrid(*load.kin_grids())
        gmask = load.mask()

        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            obs = gpath.stem.split(".")[0]
            kind = obs.split("_")[0]

            grid = pineappl.grid.Grid.read(gpath)
            pdfset = lhapdf.mkPDF(pdf)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            pred = grid.convolute_with_one(2212, pdfset.xfxQ2, pdfset.alphasQ2)
            pred = pred.reshape(xgrid.shape)
            np.save(preds_dest / obs, pred)
            ax.plot_surface(xgrid, q2grid, pred, label="yadism")

            genie_pred = genie[f"{kind}_free_p"]
            genie_pred = genie_pred[gmask].reshape(xgrid.shape)
            ax.plot_surface(xgrid, q2grid, genie_pred, label="genie")

            __import__("pdb").set_trace()
